# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT.

"""
Pipeline for Vidore v3 Evaluation using nvidia/nemotron-colembed-vl-8b-v2

This script implements a dense retrieval pipeline using NVIDIA's
nvidia/llama-nemotron-embed-vl-1b-v2 model. It demonstrates how to:
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

    pip install accelerate==1.12.0
    pip install transformers==5.0.0rc0    
    pip install flash-attn==2.6.3 --no-build-isolation
    pip install datasets==4.5.0

Usage:
    vidore-benchmark pipeline evaluate \
        --dataset-name vidore/vidore_v3_hr \
        --module-path example_pipelines/nemotron_colembed_8b_v2.py \
        --class-name NemotronColEmbed8BPipeline \
        --pipeline-args '{"batch_size": 32, "top_k": 100}' \
        --language english    
"""

import sys
import time
from typing import Any, Dict, List, Tuple
from collections import OrderedDict

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

class NemotronColEmbed8B():
    """
    Encapsulates logic for the Nemotron Embed VL model
    """    
    def __init__(self, model_name = "nvidia/nemotron-colembed-vl-8b-v2", batch_size: int = 32):
        """
        Initialize the pipeline.

        Args:
            batch_size: Number of items to process per GPU batch
        """
        self.batch_size = batch_size
        self.device = "cuda"

        # Check CUDA availability - required for this pipeline
        if not torch.cuda.is_available():
            print("Error: CUDA is not available. This pipeline requires a GPU.")
            print("Please ensure you have:")
            print("  - An NVIDIA GPU with CUDA support")
            print("  - CUDA toolkit installed")
            print("  - PyTorch with CUDA support: pip install torch --index-url https://download.pytorch.org/whl/cu118")
            sys.exit(1)

        self.model_name = model_name

        print("Initializing model on GPU...")
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
                batch_embeddings = self.model.forward_images(images=batch)

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

        # Pad all batches to the same sequence length before concatenating
        # Each batch may have different seq_len: [batch_size, seq_len, embed_dim]
        max_seq_len = max(emb.shape[1] for emb in corpus_embeddings)
        print(f"  Padding corpus embeddings to max sequence length: {max_seq_len}")

        padded_embeddings = []
        for emb in corpus_embeddings:
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
        print(f"Corpus embedding complete. Shape: {all_embeddings.shape}, Device: {all_embeddings.device}")

        return all_embeddings

    def _embed_queries_batched(self, queries: List[str]) -> torch.Tensor:
        """
        Embed query texts in batches on GPU, return on CPU.

        Args:
            queries: List of query text strings

        Returns:
            Tensor of shape [num_queries, embed_dim] on CPU
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
                batch_embeddings = self.model.forward_queries(batch)

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

class NemotronColEmbed8BPipeline(BasePipeline):
    """
    Dense retrieval pipeline using NVIDIA NeMo Retriever ColEmbed.

    This pipeline implements a memory-efficient approach:
    1. Embed corpus images on GPU in batches
    2. Move embeddings to CPU immediately to save GPU memory
    3. Embed queries on GPU
    4. Implement ColBERT-style late-interaction scoring on CPU

    This approach allows handling large corpora that wouldn't fit in GPU memory
    while maintaining reasonable scoring performance on CPU.
    """

    def __init__(self, model_name = "nvidia/nemotron-colembed-vl-8b-v2", batch_size: int = 32, top_k: int = 100):
        self.top_k = top_k
        self.batch_size = batch_size
        self.embedding_model = NemotronColEmbed8B(model_name=model_name, batch_size=batch_size)

    def index(self, corpus_ids, corpus_images, corpus_texts, dataset_name = None):
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
        # Embeds corpus and save the multi-vectors in CPU memory
        self.corpus_embeddings = self.embedding_model._embed_corpus_batched(corpus_images)
        
    def search(self, query_ids: List[str], queries: List[str]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
        """
        Retrieve relevant documents with dense retrieval.

        This method:
        1. Embeds all queries
        2. Computes late-interaction similarity scores and retrieves top candidates

        Args:
            query_ids: List of query identifiers
            queries: List of query texts

        Returns:
            Tuple of (results_dict, infos_dict) where:
            - results_dict: Dictionary mapping query_id to {corpus_id: score}
            - infos_dict: Dictionary with timing and configuration metrics
        """
        # Step 1: Embed queries (GPU → CPU)
        query_embed_start = time.time()
        print(f"\nEmbedding {len(queries)} queries...")
        query_embeddings = self.embedding_model._embed_queries_batched(queries)
        query_embed_time = time.time() - query_embed_start
        print(f"Query embedding complete in {query_embed_time:.2f} seconds. Shape: {query_embeddings.shape}")

        # Step 2: Compute scores (CPU)
        print("\nComputing similarity scores...")
        retrieve_start = time.time()
        scores = self.embedding_model._compute_maxsim_scores(query_embeddings, self.corpus_embeddings)
        print(f"Similarity computation complete. Score range: [{scores.min():.4f}, {scores.max():.4f}]")

        # Extract top-k results per query
        print(f"\nExtracting top-{self.top_k} results per query...")
        results = dict()

        # Retrieving top-k corpus items per query using the embedding model
        for q_idx, query_id in enumerate(query_ids):
            # Get scores for this query
            query_scores = scores[q_idx]

            # Get top-k indices and scores
            topk_scores, topk_indices = torch.topk(query_scores, min(self.top_k, len(self.corpus_ids)))

            # Build results dictionary
            topk_corpus_ids = OrderedDict()
            for idx, score in zip(topk_indices, topk_scores):
                topk_corpus_ids[self.corpus_ids[idx.item()]] = score.item()
            results[query_id] = topk_corpus_ids

        retrieve_time = time.time() - retrieve_start
        print(f"\nRetrieval complete in {retrieve_time:.2f} seconds")
        print(f"Average time per query: {retrieve_time / len(query_ids):.2f} seconds")
        total_time = time.time() - query_embed_start

        # Build info dictionary with metrics
        infos = {
            "query_embed_time_ms": query_embed_time * 1000,
            "retrieve_time_ms": retrieve_time * 1000,
            "total_search_time_ms": total_time * 1000,
            "retriever_model": self.embedding_model.model_name,
            "device": self.embedding_model.device,
            "batch_size": self.batch_size,
            "retriever_top_k": self.top_k
        }

        return results, infos
