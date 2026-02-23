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
from typing import Any, Dict, List, Optional, Tuple

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
        batch_size: int = 8,
        scoring_batch_size: int = 128,
        rerank_batch_size: int = 8,
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
        embeddings = self.retriever.encode_image(
            images=images,
            task="retrieval",
            batch_size=self.batch_size,
        )

        return torch.tensor(embeddings)

    def _embed_queries(self, queries: List[str]) -> torch.Tensor:
        """
        Embed text queries using JinaV4.

        Args:
            queries: List of query strings

        Returns:
            Tensor of embeddings
        """
        # Use jina-embeddings-v4 encode_text API with retrieval task
        embeddings = self.retriever.encode_text(
            texts=queries,
            task="retrieval",
            prompt_name="query",
            batch_size=self.batch_size,
        )

        return torch.tensor(embeddings)

    def _compute_similarity(self, query_embeddings: torch.Tensor, corpus_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between queries and corpus in batches.

        Args:
            query_embeddings: [num_queries, embed_dim]
            corpus_embeddings: [num_corpus, embed_dim]

        Returns:
            scores: [num_queries, num_corpus] tensor of similarity scores
        """
        # Normalize embeddings for cosine similarity
        query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
        corpus_embeddings = torch.nn.functional.normalize(corpus_embeddings, p=2, dim=1)

        num_queries = query_embeddings.shape[0]
        num_corpus = corpus_embeddings.shape[0]

        # Process in batches to manage memory
        all_scores = []

        for q_start in range(0, num_queries, self.scoring_batch_size):
            q_end = min(q_start + self.scoring_batch_size, num_queries)
            query_batch = query_embeddings[q_start:q_end]

            batch_scores = []
            for c_start in range(0, num_corpus, self.scoring_batch_size):
                c_end = min(c_start + self.scoring_batch_size, num_corpus)
                corpus_batch = corpus_embeddings[c_start:c_end]

                # Compute cosine similarity for this batch
                scores_chunk = torch.mm(query_batch, corpus_batch.T)
                batch_scores.append(scores_chunk)

            # Concatenate corpus dimension
            all_scores.append(torch.cat(batch_scores, dim=1))

        # Concatenate query dimension
        scores = torch.cat(all_scores, dim=0)

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

    def retrieve(
        self,
        query_ids: List[str],
        queries: List[str],
        corpus_ids: List[str],
        corpus_images: List[Any],
        corpus_texts: List[str],
    ) -> Tuple[Dict[str, Dict[str, float]], Optional[Dict[str, Any]]]:
        """
        Retrieve relevant documents using two-stage retrieval + visual reranking.

        This method:
        1. Embeds all corpus images using JinaV4
        2. Embeds all queries using JinaV4
        3. Computes similarity scores and retrieves top candidates
        4. Reranks candidates using Jina Visual Reranker
        5. Returns final results

        Args:
            query_ids: List of query identifiers
            queries: List of query texts
            corpus_ids: List of corpus item identifiers
            corpus_images: List of PIL.Image objects
            corpus_texts: List of markdown text strings (not used in this vision pipeline)

        Returns:
            Tuple of (results_dict, infos_dict) where:
            - results_dict: Dictionary mapping query_id to {corpus_id: score}
            - infos_dict: Dictionary with timing and configuration metrics
        """
        start_time = time.time()

        # Step 1: Embed corpus images
        print(f"\nEmbedding {len(corpus_images)} corpus images...")
        corpus_embed_start = time.time()
        corpus_embeddings = self._embed_images(corpus_images)
        corpus_embed_time = time.time() - corpus_embed_start
        print(f"Corpus embedding complete. Shape: {corpus_embeddings.shape}")

        # Step 2: Embed queries
        print(f"\nEmbedding {len(queries)} queries...")
        query_embed_start = time.time()
        query_embeddings = self._embed_queries(queries)
        query_embed_time = time.time() - query_embed_start
        print(f"Query embedding complete. Shape: {query_embeddings.shape}")

        # Step 3: Compute similarity scores
        print("\nComputing similarity scores...")
        similarity_start = time.time()
        scores = self._compute_similarity(query_embeddings, corpus_embeddings)
        similarity_time = time.time() - similarity_start
        print(f"Similarity computation complete. Score range: [{scores.min():.4f}, {scores.max():.4f}]")

        # Step 4: Retrieve top candidates and rerank
        print(f"\nReranking top-{self.retriever_top_k} candidates per query...")
        rerank_start = time.time()
        results = {}

        for q_idx, query_id in enumerate(query_ids):
            # Get top-k candidates from retrieval stage
            query_scores = scores[q_idx]
            topk_scores, topk_indices = torch.topk(query_scores, min(self.retriever_top_k, len(corpus_ids)))

            # Get candidate images and IDs
            candidate_ids = [corpus_ids[idx.item()] for idx in topk_indices]
            candidate_images_batch = [corpus_images[idx.item()] for idx in topk_indices]

            # Rerank candidates
            reranked_results = self._rerank_candidates(queries[q_idx], candidate_images_batch, candidate_ids)

            # Keep top final_top_k results
            sorted_results = sorted(reranked_results.items(), key=lambda x: x[1], reverse=True)[: self.final_top_k]
            results[query_id] = dict(sorted_results)

            if (q_idx + 1) % 10 == 0:
                print(f"  Processed {q_idx + 1}/{len(query_ids)} queries...")

        rerank_time = time.time() - rerank_start
        total_time = time.time() - start_time

        print(f"\nRetrieval and reranking complete in {total_time:.2f} seconds")

        # Build info dictionary with metrics
        infos = {
            "corpus_embed_time_ms": corpus_embed_time * 1000,
            "query_embed_time_ms": query_embed_time * 1000,
            "similarity_time_ms": similarity_time * 1000,
            "rerank_time_ms": rerank_time * 1000,
            "total_time_ms": total_time * 1000,
            "retriever_model": "jinaai/jina-embeddings-v4",
            "reranker_model": "jinaai/jina-reranker-m0",
            "device": self.device,
            "batch_size": self.batch_size,
            "scoring_batch_size": self.scoring_batch_size,
            "rerank_batch_size": self.rerank_batch_size,
            "retriever_top_k": self.retriever_top_k,
            "final_top_k": self.final_top_k,
            "num_queries": len(query_ids),
            "corpus_size": len(corpus_ids),
        }

        return results, infos
