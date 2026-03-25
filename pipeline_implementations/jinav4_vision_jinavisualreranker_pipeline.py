#!/usr/bin/env python3
"""
JinaV4 Vision + Jina Visual Reranker Pipeline for Vidore v3 Evaluation

This script implements a two-stage retrieval pipeline using:
1. jinaai/jina-embeddings-v4 for dense image retrieval on corpus images
2. jinaai/jina-reranker-m0 for visual reranking the top candidates

Pipeline stages:
1. Retrieval: Embed corpus images and queries using JinaV4, compute similarity scores
2. Reranking: Rerank top-k candidates using Jina visual reranker

Dependencies:
    pip install transformers torch pillow

Usage:
    python example_pipelines/jinav4_vision_jinavisualreranker_pipeline.py --dataset vidore/vidore_v3_computer_science
    python example_pipelines/jinav4_vision_jinavisualreranker_pipeline.py --dataset vidore/vidore_v3_industrial
"""

import sys
import time
from typing import Any, Dict, List, Tuple

try:
    import torch
    from transformers import AutoModel
except ImportError:
    print("Error: Required dependencies not installed.")
    print("Please install: pip install transformers torch pillow")
    sys.exit(1)

try:
    from vidore_benchmark import BasePipeline
except ImportError:
    print("Error: vidore_benchmark package not found.")
    print("Please install the package first: pip install -e .")
    print("Run this from the repository root directory.")
    sys.exit(1)


class JinaV4VisionJinaVisualRerankerPipeline(BasePipeline):
    """
    Two-stage retrieval pipeline using JinaV4 for image retrieval and Jina Visual Reranker.

    This pipeline operates on the image corpus (corpus_images):
    1. First stage: Dense retrieval using JinaV4 image embeddings
    2. Second stage: Visual reranking using jinaai/jina-reranker-m0

    The pipeline is optimized for visual document retrieval where the corpus
    consists of document images.
    """

    def __init__(
        self,
        batch_size: int = 1,
        scoring_batch_size: int = 128,
        rerank_batch_size: int = 1,
        retriever_top_k: int = 100,
        final_top_k: int = 100,
        device: str = "auto",
    ):
        """
        Initialize the JinaV4 Vision + Jina Visual Reranker pipeline.

        Args:
            batch_size: Number of items to process per batch for embedding
            scoring_batch_size: Number of items to process per batch for scoring
            rerank_batch_size: Number of query-image pairs to process per batch for reranking
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

        print(f"Initializing JinaV4 Vision + Jina Visual Reranker pipeline on {self.device}...")

        # Initialize the retriever model (JinaV4 for image embeddings)
        print("Loading jinaai/jina-embeddings-v4 retriever model...")
        self.retriever = AutoModel.from_pretrained(
            "jinaai/jina-embeddings-v4",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(self.device)
        self.retriever.eval()
        print("Retriever model loaded successfully!")

        # Initialize the reranker model (Jina Visual Reranker)
        print("Loading jinaai/jina-reranker-m0 reranker model...")
        try:
            self.reranker = AutoModel.from_pretrained(
                "jinaai/jina-reranker-m0",
                torch_dtype="auto",
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            ).to(self.device)
        except Exception:
            print("Flash attention not available, falling back to eager attention...")
            self.reranker = AutoModel.from_pretrained(
                "jinaai/jina-reranker-m0",
                torch_dtype="auto",
                trust_remote_code=True,
            ).to(self.device)
        self.reranker.eval()
        print("Reranker model loaded successfully!")

    def _embed_images(self, images: List[Any]) -> torch.Tensor:
        """
        Embed images using JinaV4.

        Args:
            images: List of PIL.Image objects

        Returns:
            Tensor of embeddings
        """
        # Use jina-embeddings-v4 encode_image API with retrieval task
        embeddings = []
        for image in images:
            embedding = self.retriever.encode_image(
                images=[image],
                task="retrieval",
                # batch_size=self.batch_size,
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

        assert len(embeddings) == len(images), "Number of embeddings must match number of images"

        return embeddings

    def _embed_queries(self, queries: List[str]) -> torch.Tensor:
        """
        Embed text queries using JinaV4.

        Args:
            queries: List of query strings

        Returns:
            List of tensors, each of shape [num_tokens, embed_dim] for each query
        """
        # Use jina-embeddings-v4 encode_text API with retrieval task
        embeddings = []
        for query in queries:
            embedding = self.retriever.encode_text(
                texts=[query],
                task="retrieval",
                prompt_name="query",
                # batch_size=self.batch_size,
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

        assert len(embeddings) == len(queries), "Number of embeddings must match number of queries"

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
        candidate_images: List[Any],
        candidate_ids: List[str],
    ) -> Dict[str, float]:
        """
        Rerank candidate documents using Jina Visual Reranker with batched inference.

        Args:
            query: Query text
            candidate_images: List of candidate PIL.Image objects
            candidate_ids: List of candidate document IDs

        Returns:
            Dictionary mapping corpus_id to reranked score
        """
        if not candidate_images:
            return {}

        # Create query-image pairs for the visual reranker
        image_pairs = [[query, img] for img in candidate_images]

        # Get reranker scores using compute_score with doc_type="image" in batches
        all_scores = []
        for i in range(0, len(image_pairs), self.rerank_batch_size):
            batch = image_pairs[i : i + self.rerank_batch_size]
            batch_scores = self.reranker.compute_score(batch, max_length=2048, doc_type="image")
            if hasattr(batch_scores, "__iter__"):
                all_scores.extend(batch_scores)
            else:
                all_scores.append(batch_scores)

        # Create result dictionary
        results = {cid: float(score) for cid, score in zip(candidate_ids, all_scores)}

        return results

    def index(self, corpus_ids, corpus_images, corpus_texts, dataset_name: str = None):
        """
        Store corpus data for use in search().

        This pipeline does not require additional indexing or preprocessing steps,
        so we simply store the corpus data for use during retrieval.

        Args:
            corpus_ids: List of corpus item identifiers
            corpus_images: List of PIL.Image objects
            corpus_texts: List of markdown text strings (not used in this vision pipeline)
        """
        self.corpus_ids = corpus_ids
        self.corpus_images = corpus_images

        # Step 1: Embed corpus images
        print(f"\nEmbedding {len(corpus_images)} corpus images...")
        corpus_embed_start = time.time()
        self.corpus_embeddings = self._embed_images(corpus_images)
        corpus_embed_time = time.time() - corpus_embed_start
        print(
            f"Corpus embedding complete. Shape: {self.corpus_embeddings.shape}"
            f", Time taken: {corpus_embed_time:.2f} seconds"
        )

    def search(self, query_ids: List[str], queries: List[str]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
        """
        Retrieve relevant documents using two-stage retrieval + visual reranking.

        This method:
        1. Embeds all queries using JinaV4
        2. Computes similarity scores and retrieves top candidates
        3. Reranks candidates using Jina Visual Reranker
        4. Returns final results

        Args:
            query_ids: List of query identifiers
            queries: List of query texts

        Returns:
            Tuple of (results_dict, infos_dict) where:
            - results_dict: Dictionary mapping query_id to {corpus_id: score}
            - infos_dict: Dictionary with timing and configuration metrics
        """

        # Step 1: Embed queries
        print(f"\nEmbedding {len(queries)} queries...")
        query_embed_start = time.time()
        query_embeddings = self._embed_queries(queries)
        query_embed_time = time.time() - query_embed_start
        print(f"Query embedding complete. Shape: {query_embeddings.shape}")

        # Step 2: Compute similarity scores
        print("\nComputing similarity scores...")
        similarity_start = time.time()
        scores = self._compute_similarity(query_embeddings, self.corpus_embeddings)
        similarity_time = time.time() - similarity_start
        print(f"Similarity computation complete. Score range: [{scores.min():.4f}, {scores.max():.4f}]")

        # Step 3: Retrieve top candidates and rerank
        print(f"\nReranking top-{self.retriever_top_k} candidates per query...")
        rerank_start = time.time()
        results = {}

        for q_idx, query_id in enumerate(query_ids):
            # Get top-k candidates from retrieval stage
            query_scores = scores[q_idx]
            topk_scores, topk_indices = torch.topk(query_scores, min(self.retriever_top_k, len(self.corpus_ids)))

            # Get candidate images and IDs
            candidate_ids = [self.corpus_ids[idx.item()] for idx in topk_indices]
            candidate_images_batch = [self.corpus_images[idx.item()] for idx in topk_indices]

            # Rerank candidates
            reranked_results = self._rerank_candidates(queries[q_idx], candidate_images_batch, candidate_ids)

            # Keep top final_top_k results
            sorted_results = sorted(reranked_results.items(), key=lambda x: x[1], reverse=True)[: self.final_top_k]
            results[query_id] = dict(sorted_results)

            if (q_idx + 1) % 10 == 0:
                print(f"  Processed {q_idx + 1}/{len(query_ids)} queries...")

        rerank_time = time.time() - rerank_start
        total_time = time.time() - query_embed_start

        print(f"\nRetrieval and reranking complete in {total_time:.2f} seconds")

        # Build info dictionary with metrics
        infos = {
            "query_embed_time_ms": query_embed_time * 1000,
            "similarity_time_ms": similarity_time * 1000,
            "rerank_time_ms": rerank_time * 1000,
            "total_search_time_ms": total_time * 1000,
            "retriever_model": "jinaai/jina-embeddings-v4",
            "reranker_model": "jinaai/jina-reranker-m0",
            "device": self.device,
            "batch_size": self.batch_size,
            "scoring_batch_size": self.scoring_batch_size,
            "rerank_batch_size": self.rerank_batch_size,
            "retriever_top_k": self.retriever_top_k,
            "final_top_k": self.final_top_k,
            "num_queries": len(query_ids),
            "corpus_size": len(self.corpus_ids),
        }

        return results, infos
