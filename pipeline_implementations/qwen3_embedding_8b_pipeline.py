#!/usr/bin/env python3
"""
Qwen3-Embedding-8B Pipeline for Vidore v3 Evaluation

This script implements a dense retrieval pipeline using:
- Qwen/Qwen3-Embedding-8B for text embedding retrieval

Pipeline stages:
1. Retrieval: Embed corpus texts and queries, compute cosine similarity scores

Dependencies:
    pip install sentence-transformers>=2.7.0 transformers>=4.51.0 torch

Usage:
    python example_pipelines/qwen3_embedding_8b_pipeline.py --dataset vidore/vidore_v3_computer_science
    python example_pipelines/qwen3_embedding_8b_pipeline.py --dataset vidore/vidore_v3_industrial
"""

import sys
import time
from typing import Any, Dict, List, Tuple

try:
    import torch
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: Required dependencies not installed.")
    print("Please install: pip install sentence-transformers>=2.7.0 transformers>=4.51.0 torch")
    sys.exit(1)

try:
    from vidore_benchmark import BasePipeline
except ImportError:
    print("Error: vidore_benchmark package not found.")
    print("Please install the package first: pip install -e .")
    print("Run this from the repository root directory.")
    sys.exit(1)


class Qwen3Embedding8BPipeline(BasePipeline):
    """
    Dense retrieval pipeline using Qwen3-Embedding-8B.

    This pipeline operates on the text corpus (corpus_texts) using dense embeddings
    and cosine similarity scoring for retrieval. No reranking stage is used.
    """

    def __init__(
        self,
        batch_size: int = 1,
        scoring_batch_size: int = 8,
        top_k: int = 100,
        device: str = "auto",
        use_flash_attention: bool = True,
    ):
        """
        Initialize the Qwen3-Embedding-8B pipeline.

        Args:
            batch_size: Number of items to process per batch for embedding
            scoring_batch_size: Number of items to process per batch for scoring
            top_k: Number of results to return per query
            device: Device to use ('auto', 'cuda', 'cpu', or 'mps')
            use_flash_attention: Whether to use flash attention for acceleration
        """
        self.batch_size = batch_size
        self.scoring_batch_size = scoring_batch_size
        self.top_k = top_k

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

        print(f"Initializing Qwen3-Embedding-8B pipeline on {self.device}...")

        # Initialize the embedding model
        print("Loading Qwen/Qwen3-Embedding-8B model...")
        if use_flash_attention and self.device == "cuda":
            try:
                self.model = SentenceTransformer(
                    "Qwen/Qwen3-Embedding-8B",
                    model_kwargs={
                        "attn_implementation": "flash_attention_2",
                        "device_map": "auto",
                        "torch_dtype": "torch.bfloat16",
                    },
                    tokenizer_kwargs={"padding_side": "left"},
                )
                print("Model loaded with flash attention!")
            except Exception:
                print("Flash attention not available, falling back to standard attention...")
                self.model = SentenceTransformer("Qwen/Qwen3-Embedding-8B")
        else:
            self.model = SentenceTransformer("Qwen/Qwen3-Embedding-8B")
        print("Model loaded successfully!")

    def _embed_documents(self, texts: List[str]) -> torch.Tensor:
        """
        Embed documents using Qwen3-Embedding-8B.

        Args:
            texts: List of document text strings

        Returns:
            Tensor of embeddings
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
        )

        return embeddings

    def _embed_queries(self, queries: List[str]) -> torch.Tensor:
        """
        Embed queries using Qwen3-Embedding-8B with query prompt.

        Args:
            queries: List of query strings

        Returns:
            Tensor of embeddings
        """
        embeddings = self.model.encode(
            queries,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            prompt_name="query",
        )

        return embeddings

    def _compute_similarity(self, query_embeddings: torch.Tensor, corpus_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between queries and corpus in batches.

        Args:
            query_embeddings: [num_queries, embed_dim]
            corpus_embeddings: [num_corpus, embed_dim]

        Returns:
            scores: [num_queries, num_corpus] tensor of similarity scores
        """
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

                # Use sentence-transformers similarity function for this batch
                scores_chunk = self.model.similarity(query_batch, corpus_batch)
                batch_scores.append(scores_chunk)

            # Concatenate corpus dimension
            all_scores.append(torch.cat(batch_scores, dim=1))

        # Concatenate query dimension
        scores = torch.cat(all_scores, dim=0)

        return scores

    def index(self, corpus_ids: List[str], corpus_images: List[Any], corpus_texts: List[str], dataset_name = None) -> None:
        """
        Index the corpus by embedding all texts and storing them in memory.
        The embeddings are stored in self.corpus_embeddings and the corresponding IDs and texts are stored
        in self.corpus_ids and self.corpus_texts for later retrieval.
        """
        self.corpus_ids = corpus_ids
        self.corpus_texts = corpus_texts

        print(f"\nEmbedding {len(corpus_texts)} corpus documents...")
        corpus_embed_start = time.time()
        self.corpus_embeddings = self._embed_documents(corpus_texts)
        corpus_embed_time = time.time() - corpus_embed_start
        print(
            f"Corpus embedding complete. Shape: {self.corpus_embeddings.shape}."
            f" Time taken: {corpus_embed_time:.2f} seconds"
        )

    def search(self, query_ids: List[str], queries: List[str]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
        """
        Perform retrieval by embedding queries and computing similarity scores against the indexed corpus embeddings.

        Args:
            query_ids: List of query identifiers
            queries: List of query strings
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
        scoring_start = time.time()
        scores = self._compute_similarity(query_embeddings, self.corpus_embeddings)
        scoring_time = time.time() - scoring_start
        print(f"Scoring complete. Score range: [{scores.min():.4f}, {scores.max():.4f}]")

        # Step 3: Extract top-k results per query
        print(f"\nExtracting top-{self.top_k} results per query...")
        results = {}

        for q_idx, query_id in enumerate(query_ids):
            query_scores = scores[q_idx]
            topk_scores, topk_indices = torch.topk(query_scores, min(self.top_k, len(self.corpus_ids)))

            results[query_id] = {
                self.corpus_ids[idx.item()]: score.item() for idx, score in zip(topk_indices, topk_scores)
            }

        total_time = time.time() - query_embed_start
        print(f"\nRetrieval complete in {total_time:.2f} seconds")

        # Build info dictionary with metrics
        infos = {
            "query_embed_time_ms": query_embed_time * 1000,
            "scoring_time_ms": scoring_time * 1000,
            "total_search_time_ms": total_time * 1000,
            "retriever_model": "Qwen/Qwen3-Embedding-8B",
            "device": self.device,
            "batch_size": self.batch_size,
            "scoring_batch_size": self.scoring_batch_size,
            "top_k": self.top_k,
            "num_queries": len(query_ids),
            "corpus_size": len(self.corpus_ids),
        }

        return results, infos
