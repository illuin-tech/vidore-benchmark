#!/usr/bin/env python3
"""
NeMo Retriever ColEmbed Pipeline for Vidore v3 Evaluation

This script implements a late-interaction retrieval pipeline using NVIDIA's
llama-nemoretriever-colembed-1b-v1 model. It demonstrates how to:
1. Subclass BasePipeline for a custom retrieval implementation
2. Handle GPU memory constraints by computing embeddings on GPU and storing on CPU
3. Implement ColBERT-style late-interaction scoring on CPU
4. Evaluate on vidore v3 datasets

GPU Requirements:
- NVIDIA GPU with CUDA support (tested on A100 80GB)
- CUDA toolkit installed
- Sufficient GPU memory for batch processing (adjust --batch_size if needed)

Dependencies:
    pip install torch --index-url https://download.pytorch.org/whl/cu118
    pip install transformers==4.49.0
    pip install flash-attn==2.6.3

Usage:
    python scripts/nemoretriever_colembed_pipeline.py --dataset vidore/vidore_v3_computer_science
    python scripts/nemoretriever_colembed_pipeline.py --dataset vidore/vidore_v3_industrial --batch_size 2 --top_k 50
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


class ColEmbedPipeline(BasePipeline):
    """
    Late-interaction retrieval pipeline using NVIDIA NeMo Retriever ColEmbed.

    This pipeline implements a memory-efficient approach:
    1. Embed corpus images on GPU in batches
    2. Move embeddings to CPU immediately to save GPU memory
    3. Embed queries on GPU
    4. Perform ColBERT-style MaxSim scoring on CPU

    This approach allows handling large corpora that wouldn't fit in GPU memory
    while maintaining reasonable scoring performance on CPU.
    """

    def __init__(self, batch_size: int = 32, top_k: int = 100):
        """
        Initialize the ColEmbed pipeline.

        Args:
            batch_size: Number of items to process per GPU batch
            top_k: Number of top results to return per query
        """
        self.batch_size = batch_size
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
        print("Loading nvidia/llama-nemoretriever-colembed-1b-v1...")

        # Load model with GPU settings
        try:
            self.model = AutoModel.from_pretrained(
                "nvidia/llama-nemoretriever-colembed-1b-v1",
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
                "nvidia/llama-nemoretriever-colembed-1b-v1",
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
        Compute ColBERT-style MaxSim scores between queries and corpus on CPU.

        Args:
            query_embeddings: [num_queries, query_seq_len, embed_dim]
            corpus_embeddings: [num_corpus, corpus_seq_len, embed_dim]

        Returns:
            scores: [num_queries, num_corpus] tensor of similarity scores
        """
        print("\nComputing MaxSim scores on CPU...")
        print(f"  Query embeddings: {query_embeddings.shape}")
        print(f"  Corpus embeddings: {corpus_embeddings.shape}")

        num_queries = query_embeddings.shape[0]
        num_corpus = corpus_embeddings.shape[0]

        # Initialize scores tensor
        scores = torch.zeros(num_queries, num_corpus, dtype=torch.float32)

        # Process each query
        for q_idx in range(num_queries):
            if q_idx % 10 == 0:
                print(f"  Processing query {q_idx + 1}/{num_queries}...")

            # Get query embedding: [query_seq_len, embed_dim]
            q_emb = query_embeddings[q_idx]

            # Compute similarity with all corpus items
            # For each query token, find max similarity with corpus tokens
            for c_idx in range(num_corpus):
                # Get corpus embedding: [corpus_seq_len, embed_dim]
                c_emb = corpus_embeddings[c_idx]

                # Compute token-level similarities: [query_seq_len, corpus_seq_len]
                token_sims = torch.matmul(q_emb, c_emb.T)

                # MaxSim: for each query token, take max over corpus tokens, then sum
                maxsim_score = token_sims.max(dim=1)[0].sum()
                scores[q_idx, c_idx] = maxsim_score.item()

        print(f"Scoring complete. Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        return scores

    def retrieve(
        self,
        query_ids: List[str],
        queries: List[str],
        corpus_ids: List[str],
        corpus_images: List[Any],
        corpus_texts: List[Any],
    ) -> Dict[str, Dict[str, float]]:
        """
        Retrieve relevant corpus items for each query using late-interaction.

        This method:
        1. Embeds all corpus images on GPU → CPU
        2. Embeds all queries on GPU → CPU
        3. Computes MaxSim scores on CPU
        4. Returns top-k results per query

        Args:
            query_ids: List of query identifiers
            queries: List of query texts
            corpus_ids: List of corpus item identifiers
            corpus_images: List of PIL.Image objects
            corpus_texts: List of str objects

        Returns:
            Dictionary mapping query_id to {corpus_id: score} for top-k results
        """
        start_time = time.time()

        # Step 1: Embed corpus (GPU → CPU)
        corpus_embeddings = self._embed_corpus_batched(corpus_images)

        # Step 2: Embed queries (GPU → CPU)
        query_embeddings = self._embed_queries_batched(queries)

        # Step 3: Compute scores (CPU)
        scores = self._compute_maxsim_scores(query_embeddings, corpus_embeddings)

        # Step 4: Extract top-k results per query
        print(f"\nExtracting top-{self.top_k} results per query...")
        results = {}

        for q_idx, query_id in enumerate(query_ids):
            # Get scores for this query
            query_scores = scores[q_idx]

            # Get top-k indices and scores
            topk_scores, topk_indices = torch.topk(query_scores, min(self.top_k, len(corpus_ids)))

            # Build results dictionary
            results[query_id] = {corpus_ids[idx.item()]: score.item() for idx, score in zip(topk_indices, topk_scores)}

        elapsed = time.time() - start_time
        print(f"\nRetrieval complete in {elapsed:.2f} seconds")
        print(f"Average time per query: {elapsed / len(query_ids):.2f} seconds")

        return results
