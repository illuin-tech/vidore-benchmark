# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT.

"""
Pipeline for Vidore v3 Evaluation using nvidia/llama-nemotron-colembed-vl-3b-v2

This script implements a dense retrieval pipeline using NVIDIA's
nvidia/llama-nemotron-colembed-vl-3b-v22 model. It demonstrates how to:
1. Subclass BasePipeline for a custom dense retrieval implementation
2. Handle GPU memory constraints by computing embeddings on GPU in batches and storing on CPU
3. Implement ColBERT-style late-interaction scoring on CPU
4. Evaluate on vidore v3 datasets

GPU Requirements:
- NVIDIA GPU with CUDA support (tested on A100 80GB)
- CUDA toolkit installed
- Sufficient GPU memory for batch processing (adjust --batch_size if needed)

Dependencies: 
    cd vidore-benchmark/ && pip install -e .

    pip install transformers==4.49.0
    pip install flash-attn==2.6.3 --no-build-isolation
    pip install datasets==4.5.0

Usage:
    vidore-benchmark pipeline evaluate \
        --dataset-name vidore/vidore_v3_hr \
        --module-path example_pipelines/nemotron_colembed_3b_v2.py \
        --class-name ColEmbed3BPipeline \
        --pipeline-args '{"batch_size": 32, "top_k": 100}' \
        --language english    
"""


import sys
import time
from typing import Any, Dict, List

try:
    import torch
    import torch.nn.functional as F  # noqa: N812
    from transformers import AutoModel
except ImportError:
    print("Error: Required GPU dependencies not installed.")
    print("Please install: pip install torch transformers")
    print("For flash attention: pip install flash-attn>=2.6.3")
    sys.exit(1)

try:
    from vidore_benchmark import BasePipeline
except ImportError:
    print("Error: vidore_eval package not found.")
    print("Please install the package first: pip install -e .")
    print("Run this from the repository root directory.")
    sys.exit(1)


class ColEmbed3BPipeline(BasePipeline):
    """
    Late-interaction retrieval pipeline using NVIDIA NeMo Retriever ColEmbed.

    This pipeline implements a memory-efficient approach:
    1. Embed corpus images on GPU in batches
    2. Move embeddings to CPU immediately to save GPU memory
    3. Embed queries on GPU
    4. Perform ColBERT-style MaxSim scoring with batched computation

    This approach allows handling large corpora that wouldn't fit in GPU memory
    while maintaining reasonable scoring performance.
    """

    def __init__(self, model_name="nvidia/llama-nemotron-colembed-vl-3b-v2", batch_size: int = 32, scoring_batch_size: int = 32, top_k: int = 100):
        """
        Initialize the ColEmbed pipeline.

        Args:
            batch_size: Number of items to process per GPU batch for embedding
            scoring_batch_size: Number of items to process per batch for MaxSim scoring
            top_k: Number of top results to return per query
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.scoring_batch_size = scoring_batch_size
        self.top_k = top_k
        self.device = "cuda"

        # Check CUDA availability - required for this pipeline
        if not torch.cuda.is_available():
            print("Error: CUDA is not available. This pipeline requires a GPU.")
            print("Please ensure you have:")
            print("  - An NVIDIA GPU with CUDA support")
            print("  - CUDA toolkit installed")
            print("  - PyTorch with CUDA support: pip install torch --index-url https://download.pytorch.org/whl/cu118")
            sys.exit(1)

        print("Initializing ColEmbed model on GPU...")
        print(f"Loading {self.model_name}...")

        # Load model with GPU settings
        try:
            self.model = AutoModel.from_pretrained(
                self.model_name,
                device_map="cuda",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            )
            self.model.eval()
            print("Model loaded successfully!")
            print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("\nNote: flash_attention_2 requires flash-attn to be installed:")
            print("  pip install flash-attn>=2.6.3")
            print("\nRetrying without flash attention...")

            self.model = AutoModel.from_pretrained(
                self.model_name,
                device_map="cuda",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="eager",
            )
            self.model.eval()
            print("Model loaded successfully (without flash attention)!")
            print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    def _embed_corpus_batched(self, corpus: List[Any]) -> torch.Tensor:
        """
        Embed corpus images in batches on GPU, return on CPU.

        Args:
            corpus: List of PIL.Image objects

        Returns:
            Tensor of shape [num_items, seq_len, embed_dim] on CPU
        """
        print(f"\nEmbedding {len(corpus)} corpus images in batches of {self.batch_size}...")
        corpus_embeddings = []

        num_batches = (len(corpus) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(corpus), self.batch_size):
            batch_idx = i // self.batch_size + 1
            batch = corpus[i : i + self.batch_size]

            print(f"  Batch {batch_idx}/{num_batches}: Processing {len(batch)} images on GPU...")

            with torch.no_grad():
                # Embed on GPU
                batch_embeddings = self.model.forward_passages(batch, batch_size=len(batch))

                # Move to CPU immediately to free GPU memory
                batch_embeddings_cpu = batch_embeddings.cpu()
                corpus_embeddings.append(batch_embeddings_cpu)

                # Clear GPU cache
                del batch_embeddings
                torch.cuda.empty_cache()

            if batch_idx % 5 == 0:
                print(
                    f"    GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, "
                    f"{torch.cuda.memory_reserved() / 1e9:.2f} GB reserved"
                )

        # Concatenate all batches
        all_embeddings = torch.cat(corpus_embeddings, dim=0)
        print(f"Corpus embedding complete. Shape: {all_embeddings.shape}, Device: {all_embeddings.device}")

        return all_embeddings

    def _embed_queries_batched(self, queries: List[str]) -> torch.Tensor:
        """
        Embed query texts in batches on GPU, return on CPU.

        Args:
            queries: List of query text strings

        Returns:
            Tensor of shape [num_queries, seq_len, embed_dim] on CPU
        """
        print(f"\nEmbedding {len(queries)} queries in batches of {self.batch_size}...")
        query_embeddings = []

        num_batches = (len(queries) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(queries), self.batch_size):
            batch_idx = i // self.batch_size + 1
            batch = queries[i : i + self.batch_size]

            print(f"  Batch {batch_idx}/{num_batches}: Processing {len(batch)} queries on GPU...")

            with torch.no_grad():
                # Embed on GPU
                batch_embeddings = self.model.forward_queries(batch, batch_size=len(batch))

                # Move to CPU
                batch_embeddings_cpu = batch_embeddings.cpu()
                query_embeddings.append(batch_embeddings_cpu)

                # Clear GPU cache
                del batch_embeddings
                torch.cuda.empty_cache()

        # Pad all batches to the same sequence length before concatenating
        # Each batch may have different seq_len: [batch_size, seq_len, embed_dim]
        max_seq_len = max(emb.shape[1] for emb in query_embeddings)
        print(f"  Padding query embeddings to max sequence length: {max_seq_len}")

        padded_embeddings = []
        for emb in query_embeddings:
            if emb.shape[1] < max_seq_len:
                # Pad along dimension 1 (sequence length)
                # F.pad format: (left, right, top, bottom, front, back) for last dims
                # We want to pad dim 1, so: (dim2_left, dim2_right, dim1_left, dim1_right, ...)
                pad_len = max_seq_len - emb.shape[1]
                padded = F.pad(emb, (0, 0, 0, pad_len), mode="constant", value=0)
                padded_embeddings.append(padded)
            else:
                padded_embeddings.append(emb)

        # Concatenate all batches
        all_embeddings = torch.cat(padded_embeddings, dim=0)
        print(f"Query embedding complete. Shape: {all_embeddings.shape}, Device: {all_embeddings.device}")

        return all_embeddings

    def _compute_maxsim_scores(self, query_embeddings: torch.Tensor, corpus_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute ColBERT-style MaxSim scores between queries and corpus with batched computation.

        Args:
            query_embeddings: [num_queries, query_seq_len, embed_dim]
            corpus_embeddings: [num_corpus, corpus_seq_len, embed_dim]

        Returns:
            scores: [num_queries, num_corpus] tensor of similarity scores
        """
        print("\nComputing MaxSim scores with batched computation...")
        print(f"  Query embeddings: {query_embeddings.shape}")
        print(f"  Corpus embeddings: {corpus_embeddings.shape}")

        num_queries = query_embeddings.shape[0]
        num_corpus = corpus_embeddings.shape[0]

        # Initialize scores tensor
        scores: List[torch.Tensor] = []

        # Process queries in batches
        for q_start in range(0, num_queries, self.scoring_batch_size):
            q_end = min(q_start + self.scoring_batch_size, num_queries)
            query_batch = query_embeddings[q_start:q_end]  # [batch_q, q_seq, embed_dim]

            if q_start % (self.scoring_batch_size * 5) == 0:
                print(f"  Processing queries {q_start + 1}-{q_end}/{num_queries}...")

            batch_scores = []

            # Process corpus in batches
            for c_start in range(0, num_corpus, self.scoring_batch_size):
                c_end = min(c_start + self.scoring_batch_size, num_corpus)
                corpus_batch = corpus_embeddings[c_start:c_end]  # [batch_c, c_seq, embed_dim]

                # Compute token-level similarities: [batch_q, batch_c, q_seq, c_seq]
                # Using einsum: for each query token and corpus token, compute dot product
                token_sims = torch.einsum("bnd,csd->bcns", query_batch, corpus_batch)

                # MaxSim: max over corpus tokens (dim=3), then sum over query tokens (dim=2)
                chunk_scores = token_sims.max(dim=3)[0].sum(dim=2)  # [batch_q, batch_c]
                batch_scores.append(chunk_scores)

            # Concatenate corpus dimension
            scores.append(torch.cat(batch_scores, dim=1))

        # Concatenate query dimension
        all_scores = torch.cat(scores, dim=0)
        print(f"Scoring complete. Score range: [{all_scores.min():.4f}, {all_scores.max():.4f}]")

        return all_scores

    def index(
        self,
        corpus_ids: List[str],
        corpus_images: List[Any],
        corpus_texts: List[Any],
    ) -> Dict[str, Any]:
        """
        Precompute and store corpus embeddings for later search.

        Returns:
            Dictionary containing indexed data needed for search.
        """
        start_time = time.time()
        corpus_embeddings = self._embed_corpus_batched(corpus_images)
        elapsed = time.time() - start_time
        print(f"\nIndexing complete in {elapsed:.2f} seconds")

        self.corpus_ids = corpus_ids
        self.corpus_embeddings = corpus_embeddings

    def search(
        self,
        query_ids: List[str],
        queries: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        Search indexed corpus for each query.

        Args:
            query_ids: List of query identifiers
            queries: List of query texts

        Returns:
            Dictionary mapping query_id to {corpus_id: score} for top-k results
        """
        start_time = time.time()
        query_embeddings = self._embed_queries_batched(queries)
        embed_query_time = time.time() - start_time
        print(f"\nQuery embedding complete in {embed_query_time:.2f} seconds")
        scores = self._compute_maxsim_scores(query_embeddings, self.corpus_embeddings)        
        retrieval_time = time.time()
        print(f"\nExtracting top-{self.top_k} results per query...")
        results = {}

        for q_idx, query_id in enumerate(query_ids):
            # Get scores for this query
            query_scores = scores[q_idx]

            # Get top-k indices and scores
            topk_scores, topk_indices = torch.topk(query_scores, min(self.top_k, len(self.corpus_ids)))

            # Build results dictionary
            results[query_id] = {
                self.corpus_ids[idx.item()]: score.item() for idx, score in zip(topk_indices, topk_scores)
            }

        retrieval_time = time.time() - retrieval_time
        print(f"Relevance scoring complete in {retrieval_time:.2f} seconds")

        elapsed = time.time() - start_time
        print(f"\nRetrieval complete in {elapsed:.2f} seconds")
        print(f"Average time per query: {elapsed / len(query_ids):.2f} seconds")

        additional_info = {
            "embedding_time_ms": embed_query_time * 1000,
            "retrieval_time": retrieval_time * 1000,
            "total_search_time_ms": elapsed * 1000,
        }

        return results, additional_info
