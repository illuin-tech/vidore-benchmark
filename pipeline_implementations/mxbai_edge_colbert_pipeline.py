#!/usr/bin/env python3
"""
MixedBread mxbai-edge-colbert-v0-32m Pipeline for Vidore v3 Evaluation

This script implements a ColBERT retrieval pipeline using:
- mixedbread-ai/mxbai-edge-colbert-v0-32m for late-interaction retrieval on text corpus

Pipeline stages:
1. Retrieval: Embed corpus texts and queries using ColBERT, compute MaxSim scores

Dependencies:
    pip install pylate torch

Usage:
    python example_pipelines/mxbai_edge_colbert_pipeline.py --dataset vidore/vidore_v3_computer_science
    python example_pipelines/mxbai_edge_colbert_pipeline.py --dataset vidore/vidore_v3_industrial
"""

import sys
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    from pylate import models
except ImportError:
    print("Error: Required dependencies not installed.")
    print("Please install: pip install pylate torch")
    sys.exit(1)

try:
    from vidore_benchmark import BasePipeline
except ImportError:
    print("Error: vidore_benchmark package not found.")
    print("Please install the package first: pip install -e .")
    print("Run this from the repository root directory.")
    sys.exit(1)


class MxbaiEdgeColbertPipeline(BasePipeline):
    """
    ColBERT retrieval pipeline using mixedbread-ai/mxbai-edge-colbert-v0-32m.

    This pipeline operates on the text corpus (corpus_texts) using late-interaction
    ColBERT scoring (MaxSim) for retrieval. No reranking stage is used.
    """

    def __init__(
        self,
        batch_size: int = 32,
        top_k: int = 100,
        device: str = "auto",
    ):
        """
        Initialize the mxbai-edge-colbert pipeline.

        Args:
            batch_size: Number of items to process per batch
            top_k: Number of results to return per query
            device: Device to use ('auto', 'cuda', 'cpu', or 'mps')
        """
        self.batch_size = batch_size
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

        print(f"Initializing mxbai-edge-colbert pipeline on {self.device}...")

        # Initialize the ColBERT model
        print("Loading mixedbread-ai/mxbai-edge-colbert-v0-32m model...")
        self.model = models.ColBERT(
            model_name_or_path="mixedbread-ai/mxbai-edge-colbert-v0-32m",
        )
        print("Model loaded successfully!")

    def _embed_documents(self, texts: List[str]) -> List[torch.Tensor]:
        """
        Embed documents using ColBERT.

        Args:
            texts: List of document text strings

        Returns:
            List of embedding tensors, each of shape (seq_len, embed_dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            is_query=False,
            show_progress_bar=True,
        )

        # Convert to list of tensors
        return [torch.tensor(emb, device=self.device) for emb in embeddings]

    def _embed_queries(self, queries: List[str]) -> List[torch.Tensor]:
        """
        Embed queries using ColBERT.

        Args:
            queries: List of query strings

        Returns:
            List of embedding tensors, each of shape (seq_len, embed_dim)
        """
        embeddings = self.model.encode(
            queries,
            batch_size=self.batch_size,
            is_query=True,
            show_progress_bar=True,
        )

        # Convert to list of tensors
        return [torch.tensor(emb, device=self.device) for emb in embeddings]

    def _compute_maxsim_scores(
        self,
        query_embeddings: List[torch.Tensor],
        document_embeddings: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute ColBERT MaxSim scores between queries and documents.

        Args:
            query_embeddings: List of query embeddings, each of shape (q_seq_len, embed_dim)
            document_embeddings: List of document embeddings, each of shape (d_seq_len, embed_dim)

        Returns:
            scores: [num_queries, num_documents] tensor of similarity scores
        """
        scores: List[torch.Tensor] = []

        for i in range(0, len(query_embeddings), self.batch_size):
            batch_scores = []
            qs_batch = torch.nn.utils.rnn.pad_sequence(
                query_embeddings[i : i + self.batch_size],
                batch_first=True,
                padding_value=0,
            )
            for j in range(0, len(document_embeddings), self.batch_size):
                ps_batch = torch.nn.utils.rnn.pad_sequence(
                    document_embeddings[j : j + self.batch_size],
                    batch_first=True,
                    padding_value=0,
                )
                # MaxSim: for each query token, take max similarity with doc tokens, then sum
                batch_scores.append(torch.einsum("bnd,csd->bcns", qs_batch, ps_batch).max(dim=3)[0].sum(dim=2))
            batch_scores = torch.cat(batch_scores, dim=1)
            scores.append(batch_scores)

        return torch.cat(scores, dim=0)

    def retrieve(
        self,
        query_ids: List[str],
        queries: List[str],
        corpus_ids: List[str],
        corpus_images: List[Any],
        corpus_texts: List[str],
    ) -> Tuple[Dict[str, Dict[str, float]], Optional[Dict[str, Any]]]:
        """
        Retrieve relevant documents using ColBERT late-interaction scoring.

        This method:
        1. Embeds all corpus texts using ColBERT
        2. Embeds all queries using ColBERT
        3. Computes MaxSim scores
        4. Returns top-k results

        Args:
            query_ids: List of query identifiers
            queries: List of query texts
            corpus_ids: List of corpus item identifiers
            corpus_images: List of PIL.Image objects (not used in this text-only pipeline)
            corpus_texts: List of markdown text strings

        Returns:
            Tuple of (results_dict, infos_dict) where:
            - results_dict: Dictionary mapping query_id to {corpus_id: score}
            - infos_dict: Dictionary with timing and configuration metrics
        """
        start_time = time.time()

        # Step 1: Embed corpus texts
        print(f"\nEmbedding {len(corpus_texts)} corpus documents...")
        corpus_embed_start = time.time()
        corpus_embeddings = self._embed_documents(corpus_texts)
        corpus_embed_time = time.time() - corpus_embed_start
        print(f"Corpus embedding complete. {len(corpus_embeddings)} embeddings generated.")

        # Step 2: Embed queries
        print(f"\nEmbedding {len(queries)} queries...")
        query_embed_start = time.time()
        query_embeddings = self._embed_queries(queries)
        query_embed_time = time.time() - query_embed_start
        print(f"Query embedding complete. {len(query_embeddings)} embeddings generated.")

        # Step 3: Compute MaxSim scores
        print("\nComputing MaxSim scores...")
        scoring_start = time.time()
        scores = self._compute_maxsim_scores(query_embeddings, corpus_embeddings)
        scoring_time = time.time() - scoring_start
        print(f"Scoring complete. Score range: [{scores.min():.4f}, {scores.max():.4f}]")

        # Step 4: Extract top-k results per query
        print(f"\nExtracting top-{self.top_k} results per query...")
        results = {}

        for q_idx, query_id in enumerate(query_ids):
            query_scores = scores[q_idx]
            topk_scores, topk_indices = torch.topk(query_scores, min(self.top_k, len(corpus_ids)))

            results[query_id] = {corpus_ids[idx.item()]: score.item() for idx, score in zip(topk_indices, topk_scores)}

        total_time = time.time() - start_time
        print(f"\nRetrieval complete in {total_time:.2f} seconds")

        # Build info dictionary with metrics
        infos = {
            "corpus_embed_time_ms": corpus_embed_time * 1000,
            "query_embed_time_ms": query_embed_time * 1000,
            "scoring_time_ms": scoring_time * 1000,
            "total_time_ms": total_time * 1000,
            "retriever_model": "mixedbread-ai/mxbai-edge-colbert-v0-32m",
            "device": self.device,
            "batch_size": self.batch_size,
            "top_k": self.top_k,
            "num_queries": len(query_ids),
            "corpus_size": len(corpus_ids),
        }

        return results, infos
