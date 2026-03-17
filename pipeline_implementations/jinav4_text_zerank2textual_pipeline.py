#!/usr/bin/env python3
"""
JinaV4 Text + ZeRank2 Reranker Pipeline for Vidore v3 Evaluation

This script implements a two-stage retrieval pipeline using:
1. jinaai/jina-embeddings-v4 for dense text retrieval on markdown corpus
2. zeroentropy/zerank2 for reranking the top candidates

Pipeline stages:
1. Retrieval: Embed corpus texts and queries using JinaV4, compute similarity scores
2. Reranking: Rerank top-k candidates using ZeRank2 cross-encoder

Dependencies:
    pip install sentence-transformers transformers torch

Usage:
    python example_pipelines/jinav4_text_zerank2textual_pipeline.py --dataset vidore/vidore_v3_computer_science
    python example_pipelines/jinav4_text_zerank2textual_pipeline.py --dataset vidore/vidore_v3_industrial
"""

import sys
import time
from typing import Dict, List

from tqdm import tqdm

try:
    import torch
    from sentence_transformers import CrossEncoder
    from transformers import AutoModel
except ImportError:
    print("Error: Required dependencies not installed.")
    print("Please install: pip install sentence-transformers transformers torch")
    sys.exit(1)

try:
    from vidore_benchmark import BasePipeline
except ImportError:
    print("Error: vidore_benchmark package not found.")
    print("Please install the package first: pip install -e .")
    print("Run this from the repository root directory.")
    sys.exit(1)


class JinaV4TextZeRank2TextualPipeline(BasePipeline):
    """
    Two-stage retrieval pipeline using JinaV4 for text retrieval and ZeRank2 for reranking.

    This pipeline operates on the markdown corpus (corpus_texts) rather than images:
    1. First stage: Dense retrieval using JinaV4 text embeddings
    2. Second stage: Cross-encoder reranking using ZeRank2

    The pipeline is optimized for textual document retrieval where the corpus
    consists of markdown representations of documents.
    """

    def __init__(
        self,
        batch_size: int = 8,
        scoring_batch_size: int = 128,
        rerank_batch_size: int = 1,
        retriever_top_k: int = 100,
        final_top_k: int = 100,
        device: str = "auto",
    ):
        """
        Initialize the JinaV4 + ZeRank2 pipeline.

        Args:
            batch_size: Number of items to process per batch for embedding
            scoring_batch_size: Number of items to process per batch for scoring
            rerank_batch_size: Number of query-document pairs to process per batch for reranking
            retriever_top_k: Number of candidates to retrieve before reranking
            final_top_k: Number of final results to return per query
            device: Device to use ('auto', 'cuda', 'cpu', or 'mps')
        """
        self.batch_size = batch_size
        self.scoring_batch_size = scoring_batch_size
        self.rerank_batch_size = rerank_batch_size
        self.retriever_top_k = retriever_top_k
        self.final_top_k = final_top_k

        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"Initializing JinaV4 + ZeRank2 pipeline on {self.device}...")

        # Initialize the retriever model (JinaV4 for text embeddings)
        print("Loading jinaai/jina-embeddings-v4 retriever model...")
        self.retriever = AutoModel.from_pretrained(
            "jinaai/jina-embeddings-v4",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(self.device)
        self.retriever.eval()
        print("Retriever model loaded successfully!")

        # Initialize the reranker model (ZeRank2)
        print("Loading zeroentropy/zerank-2 reranker model...")
        self.reranker = CrossEncoder(
            "zeroentropy/zerank-2",
            trust_remote_code=True,
            device="cuda",
            revision="refs/pr/2",
            automodel_args={
                "attn_implementation": "flash_attention_2",
                "torch_dtype": torch.bfloat16,  # Flash Attention requires half precision
            },
        )
        print("Reranker model loaded successfully!")

    def _embed_texts(self, texts: List[str], is_query: bool = False) -> List[torch.Tensor]:
        """
        Embed texts using JinaV4 with multi-vector embeddings.

        Args:
            texts: List of text strings to embed
            is_query: Whether these are queries (vs passages)

        Returns:
            List of tensors, each of shape [num_tokens, embed_dim] for each text
        """
        prompt_name = "query" if is_query else "passage"

        # Use jina-embeddings-v4 encode_text API with retrieval task
        embeddings = []
        for text in texts:
            embedding = self.retriever.encode_text(
                texts=[text],
                task="retrieval",
                prompt_name=prompt_name,
                return_multivector=True,
            )

            embeddings.append(embedding[0])

        # Pad embeddings to max length
        max_length = max(e.shape[0] for e in embeddings)
        padded_embeddings = []
        for e in embeddings:
            if e.shape[0] < max_length:
                padding = torch.zeros(max_length - e.shape[0], e.shape[1], device=e.device)
                padded = torch.cat([e, padding], dim=0)
            else:
                padded = e
            padded_embeddings.append(padded)

        embeddings = torch.stack(padded_embeddings).to(self.device)
        # Returns list of tensors, each [num_tokens, embed_dim]
        return embeddings

    def _compute_similarity(
        self, query_embeddings: List[torch.Tensor], corpus_embeddings: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute max similarity between multi-vector queries and corpus.

        For multi-vector embeddings, the similarity between a query and a document
        is computed as the average of the maximum cosine similarity for each query
        token across all document tokens (ColBERT-style late interaction).

        Args:
            query_embeddings: List of [num_query_tokens, embed_dim] tensors
            corpus_embeddings: List of [num_doc_tokens, embed_dim] tensors

        Returns:
            scores: [num_queries, num_corpus] tensor of max similarity scores
        """
        num_queries = len(query_embeddings)
        num_corpus = len(corpus_embeddings)

        # Normalize all embeddings for cosine similarity
        query_embeddings_norm = [torch.nn.functional.normalize(q, p=2, dim=1) for q in query_embeddings]
        corpus_embeddings_norm = [torch.nn.functional.normalize(c, p=2, dim=1) for c in corpus_embeddings]

        # Compute max similarity for each query-corpus pair
        scores = torch.zeros(num_queries, num_corpus, device=self.device)

        for q_idx in range(num_queries):
            query_vecs = query_embeddings_norm[q_idx]  # [num_query_tokens, embed_dim]

            for c_start in range(0, num_corpus, self.scoring_batch_size):
                c_end = min(c_start + self.scoring_batch_size, num_corpus)
                batch_corpus = corpus_embeddings_norm[c_start:c_end]
                batch_size = len(batch_corpus)

                # Pad corpus embeddings to same length for batched computation
                max_doc_tokens = max(c.size(0) for c in batch_corpus)
                embed_dim = batch_corpus[0].size(1)

                # Create padded tensor and mask
                padded_corpus = torch.zeros(batch_size, max_doc_tokens, embed_dim, device=self.device)
                mask = torch.zeros(batch_size, max_doc_tokens, device=self.device, dtype=torch.bool)

                for i, c in enumerate(batch_corpus):
                    num_tokens = c.size(0)
                    padded_corpus[i, :num_tokens] = c
                    mask[i, :num_tokens] = True

                # Compute similarities: [batch_size, num_query_tokens, max_doc_tokens]
                sim_matrix = torch.einsum("qd,bcd->bqc", query_vecs, padded_corpus)

                # Apply mask: set padded positions to -inf so they don't affect max
                sim_matrix = sim_matrix.masked_fill(~mask.unsqueeze(1), float("-inf"))

                # Max over doc tokens for each query token: [batch_size, num_query_tokens]
                max_per_query_token = sim_matrix.max(dim=-1).values

                # Average over query tokens: [batch_size]
                batch_scores = max_per_query_token.mean(dim=-1)

                scores[q_idx, c_start:c_end] = batch_scores

        return scores

    def _rerank_candidates(
        self,
        query: str,
        candidate_texts: List[str],
        candidate_ids: List[str],
    ) -> Dict[str, float]:
        """
        Rerank candidate documents using ZeRank2 with batched inference.

        Args:
            query: Query text
            candidate_texts: List of candidate document texts
            candidate_ids: List of candidate document IDs

        Returns:
            Dictionary mapping corpus_id to reranked score
        """
        if not candidate_texts:
            return {}

        # Create query-document pairs for the cross-encoder
        query_document_pairs = [(query, doc) for doc in candidate_texts]

        # Get reranker scores using CrossEncoder.predict with batched inference
        all_scores = []
        for i in range(0, len(query_document_pairs), self.rerank_batch_size):
            batch = query_document_pairs[i : i + self.rerank_batch_size]
            batch_scores = self.reranker.predict(batch)
            all_scores.extend(batch_scores if hasattr(batch_scores, "__iter__") else [batch_scores])

        # Create result dictionary
        results = {cid: float(score) for cid, score in zip(candidate_ids, all_scores)}

        return results

    def index(self, corpus_ids: List[str], corpus_images: List[str], corpus_texts: List[str], dataset_name: str = None) -> None:
        """
        Indexing step for the pipeline. For this implementation, we don't need to do
        anything here since we compute embeddings on the fly in the search method.
        However, we could add caching logic here if desired.

        Args:
            corpus_ids: List of corpus item identifiers
            corpus_images: List of corpus images
            corpus_texts: List of corpus texts
        """
        self.corpus_ids = corpus_ids
        self.corpus_texts = corpus_texts

        print(f"\nEmbedding {len(corpus_texts)} corpus documents...")
        self.corpus_embeddings = self._embed_texts(corpus_texts, is_query=False)
        print(f"Corpus embedding complete. {len(self.corpus_embeddings)} multi-vector embeddings")

    def search(self, query_ids: List[str], queries: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Search method to perform retrieval and reranking.

        This method:
        1. Embeds queries using JinaV4
        2. Computes similarity scores with pre-embedded corpus
        3. Reranks top-k candidates using ZeRank2
        4. Returns final results

        Args:
            query_ids: List of query identifiers
            queries: List of query texts

        Returns:
            Dictionary mapping query_id to {corpus_id: score} pairs
        """
        print(f"\nEmbedding {len(queries)} queries...")
        query_embed_start = time.time()
        query_embeddings = self._embed_texts(queries, is_query=True)
        query_embed_time = time.time() - query_embed_start
        print(f"Query embedding complete. {len(query_embeddings)} multi-vector embeddings")

        print("\nComputing similarity scores...")
        similarity_start = time.time()
        scores = self._compute_similarity(query_embeddings, self.corpus_embeddings)
        similarity_time = time.time() - similarity_start
        print(f"Similarity computation complete. Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        self.corpus_embeddings = None  # Free memory
        query_embeddings = None  # Free memory

        print(f"\nReranking top-{self.retriever_top_k} candidates per query...")
        rerank_start = time.time()
        results = {}

        for q_idx, query_id in tqdm(enumerate(query_ids), total=len(query_ids), desc="Reranking queries"):
            # Get top-k candidates from retrieval stage
            query_scores = scores[q_idx]
            topk_scores, topk_indices = torch.topk(query_scores, min(self.retriever_top_k, len(self.corpus_ids)))

            # Get candidate texts and IDs
            candidate_ids = [self.corpus_ids[idx.item()] for idx in topk_indices]
            candidate_texts = [self.corpus_texts[idx.item()] for idx in topk_indices]

            # Rerank candidates
            reranked_results = self._rerank_candidates(queries[q_idx], candidate_texts, candidate_ids)

            # Keep top final_top_k results
            sorted_results = sorted(reranked_results.items(), key=lambda x: x[1], reverse=True)[: self.final_top_k]
            results[query_id] = dict(sorted_results)

            if (q_idx + 1) % 10 == 0:
                print(f"  Processed {q_idx + 1}/{len(query_ids)} queries...")

        rerank_time = time.time() - rerank_start
        total_time = time.time() - query_embed_start

        print(f"\nRetrieval and reranking complete in {total_time:.2f} seconds")

        additional_info = {
            "query_embed_time_ms": query_embed_time * 1000,
            "similarity_time_ms": similarity_time * 1000,
            "rerank_time_ms": rerank_time * 1000,
            "total_search_time_ms": total_time * 1000,
            "retriever_model": "jinaai/jina-embeddings-v4",
            "reranker_model": "zeroentropy/zerank-2",
            "device": self.device,
            "batch_size": self.batch_size,
            "scoring_batch_size": self.scoring_batch_size,
            "rerank_batch_size": self.rerank_batch_size,
            "retriever_top_k": self.retriever_top_k,
            "final_top_k": self.final_top_k,
            "num_queries": len(query_ids),
            "corpus_size": len(self.corpus_ids),
        }

        return results, additional_info
