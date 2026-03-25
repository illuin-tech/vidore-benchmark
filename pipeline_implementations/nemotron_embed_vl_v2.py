# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT.

"""
Pipeline for Vidore v3 Evaluation using nvidia/llama-nemotron-embed-vl-1b-v2

This script implements a dense retrieval pipeline using NVIDIA's
nvidia/llama-nemotron-embed-vl-1b-v2 model. It demonstrates how to:
1. Subclass BasePipeline for a custom dense retrieval implementation
2. Handle GPU memory constraints by computing embeddings on GPU in batches and storing on CPU
3. Evaluate on vidore v3 datasets

GPU Requirements:
- NVIDIA GPU with CUDA support (tested on A100 80GB)
- CUDA toolkit installed
- Sufficient GPU memory for batch processing (adjust --batch_size if needed)

Dependencies:
    cd vidore-benchmark/ && pip install -e .
    pip install "transformers>=4.47.1,<5.0.0"
    pip install flash-attn==2.6.3 --no-build-isolation
    pip install datasets==4.5.0

Usage:    
    vidore-benchmark pipeline evaluate \
        --dataset-name vidore/vidore_v3_hr \
        --module-path example_pipelines/nemotron_embed_vl_v2.py \
        --class-name NemotronEmbedVLPipeline \
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


def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)

def _l2_normalize_fp32(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x32 = x.to(torch.float32)
    x32 = _l2_normalize(x32, eps)
    return x32
    
class NemotronEmbedVL():
    """
    Encapsulates logic for the Nemotron Embed VL model
    """    
    def __init__(self, model_name = "nvidia/llama-nemotron-embed-vl-1b-v2", batch_size: int = 32, modality: str = "image_text"):
        """
        Initialize the pipeline.

        Args:
            batch_size: Number of items to process per GPU batch
        """
        self.batch_size = batch_size
        self.modality = modality
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
            self._set_processor_config()
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

    def _set_processor_config(self):
        """
        Sets some configuration of the processor for better accuracy
        """
        def _doc_len_by_modality(modality: str) -> int:
            m = str(modality or "").strip().lower()
            if m == "image":
                return 2048
            if m == "text":
                return 8192
            if m == "image_text":
                return 10240
            raise ValueError(f"Unknown doc_modality '{modality}'. Expected: 'image', 'text', or 'image_text'.")


        p_max_length = _doc_len_by_modality(self.modality)
        self.model.processor.p_max_length = p_max_length
        # Sets max number of tiles an image can be split. Each tile consumes 256 tokens.
        self.model.processor.max_input_tiles = 6
        # Enables an extra tile with the full image at lower resolution
        self.model.processor.use_thumbnail = True

    def _embed_corpus_batched(self, 
                              corpus_images: List[Any],
                              corpus_texts: List[str]) -> torch.Tensor:
        """
        Embed corpus images in batches on GPU, return on CPU.

        Args:
            corpus: List of PIL.Image objects

        Returns:
            Tensor of shape [num_items, seq_len, embed_dim] on CPU
        """
        corpus_size = len(corpus_images)
        print(f"\nEmbedding {corpus_size} corpus images in batches of {self.batch_size}...")
        corpus_embeddings = []

        num_batches = (corpus_size + self.batch_size - 1) // self.batch_size

        for i in range(0, corpus_size, self.batch_size):
            batch_idx = i // self.batch_size + 1
            batch_images = None
            batch_text = None
            if self.modality in ["image", "image_text"]:
                batch_images = corpus_images[i : i + self.batch_size]
                batch_images = [img.convert("RGB") for img in batch_images]
            if self.modality in ["text", "image_text"]:
                batch_text = corpus_texts[i : i + self.batch_size]

            #print(f"  Batch {batch_idx}/{num_batches}: Processing {len(batch_images)} images on GPU...")

            with torch.inference_mode():
                # Embed on GPU
                batch_embeddings = self.model.encode_documents(images=batch_images, texts=batch_text)
                batch_embeddings = _l2_normalize_fp32(batch_embeddings).to(torch.float16)

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
            Tensor of shape [num_queries, embed_dim] on CPU
        """
        print(f"\nEmbedding {len(queries)} queries in batches of {self.batch_size}...")
        query_embeddings = []

        num_batches = (len(queries) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(queries), self.batch_size):
            batch_idx = i // self.batch_size + 1
            batch = queries[i : i + self.batch_size]

            print(f"  Batch {batch_idx}/{num_batches}: Processing {len(batch)} queries on GPU...")

            with torch.inference_mode():
                # Embed on GPU
                batch_embeddings = self.model.encode_queries(batch)
                batch_embeddings = _l2_normalize_fp32(batch_embeddings).to(torch.float16)

                # Move to CPU
                batch_embeddings_cpu = batch_embeddings.cpu()
                query_embeddings.append(batch_embeddings_cpu)

                # Clear GPU cache
                del batch_embeddings
                torch.cuda.empty_cache()

        # Concatenate all batches
        all_embeddings = torch.cat(query_embeddings, dim=0)
        print(f"Query embedding complete. Shape: {all_embeddings.shape}, Device: {all_embeddings.device}")

        return all_embeddings

    def _compute_scores(self, query_embeddings: torch.Tensor, corpus_embeddings: torch.Tensor, batched_scoring = True) -> torch.Tensor:
        """
        Compute scores between queries and corpus on CPU.

        Args:
            query_embeddings: [num_queries, embed_dim]
            corpus_embeddings: [num_corpus, embed_dim]

        Returns:
            scores: [num_queries, num_corpus] tensor of similarity scores
        """
        print("\nComputing scores on CPU...")
        print(f"  Query embeddings: {query_embeddings.shape}")
        print(f"  Corpus embeddings: {corpus_embeddings.shape}")

        if not batched_scoring:
            # For small number of queries and documents you can run dot product very quickly
            scores = query_embeddings @ corpus_embeddings.T
            return scores

        num_queries = query_embeddings.shape[0]
        num_corpus = corpus_embeddings.shape[0]

        # Initialize scores tensor
        scores = torch.zeros(num_queries, num_corpus, dtype=torch.float32)        

        # Process each query
        for q_idx in range(num_queries):
            if q_idx % 10 == 0:
                print(f"  Processing query {q_idx + 1}/{num_queries}...")

            # Get query embedding:
            q_emb = query_embeddings[q_idx]

            # Compute similarity with all corpus items
            # For each query token, find max similarity with corpus tokens
            for c_idx in range(num_corpus):
                # Get corpus embedding: [corpus_seq_len, embed_dim]
                c_emb = corpus_embeddings[c_idx]

                # Compute token-level similarities: [query_seq_len, corpus_seq_len]
                chunk_scores = torch.matmul(q_emb, c_emb.T)
                scores[q_idx, c_idx] = chunk_scores.item()

        print(f"Scoring complete. Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        return scores 

class NemotronEmbedVLPipeline(BasePipeline):
    """
    Dense retrieval pipeline using NVIDIA NeMo Retriever ColEmbed.

    This pipeline implements a memory-efficient approach:
    1. Embed corpus images on GPU in batches
    2. Move embeddings to CPU immediately to save GPU memory
    3. Embed queries on GPU
    4. Perform scoring of queries over embeddings

    This approach allows handling large corpora that wouldn't fit in GPU memory
    while maintaining reasonable scoring performance on CPU.
    """

    def __init__(self, model_name = "nvidia/llama-nemotron-embed-vl-1b-v2", batch_size: int = 32, top_k: int = 100, modality: str = "image_text"):
        self.top_k = top_k
        self.modality = modality
        self.batch_size = batch_size
        self.embedding_model = NemotronEmbedVL(model_name=model_name, batch_size=batch_size, modality=self.modality)

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
        self.corpus_texts = corpus_texts

        # Step 1: Embed corpus (GPU → CPU)
        corpus_embed_start = time.time()
        print(f"\nEmbedding {len(self.corpus_images)} corpus images...")
        self.corpus_embeddings = self.embedding_model._embed_corpus_batched(self.corpus_images, self.corpus_texts)
        corpus_embed_time = time.time() - corpus_embed_start
        print(
            f"Corpus embedding complete. Shape: {self.corpus_embeddings.shape}"
            f", Time taken: {corpus_embed_time:.2f} seconds"
        )

    def search(self, query_ids: List[str], queries: List[str]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
        """
        Retrieve relevant documents using two-stage retrieval + visual reranking.

        This method:
        1. Embeds all queries using nvidia/llama-nemotron-embed-vl-1b-v2
        2. Computes similarity scores and retrieves top candidates
        3. Returns final results

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
        query_embeddings = self.embedding_model._embed_queries_batched(queries)
        query_embed_time = time.time() - query_embed_start
        print(f"Query embedding complete. Shape: {query_embeddings.shape}")

        # Step 2: Compute similarity scores
        print("\nComputing similarity scores...")
        retrieve_start = time.time()
        scores = self.embedding_model._compute_scores(query_embeddings, self.corpus_embeddings, batched_scoring=False)

        # Step 3: Extract top-k results per query
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
        print(f"Similarity computation complete. Score range: [{scores.min():.4f}, {scores.max():.4f}]")

        total_time = time.time() - query_embed_start
        print(f"\nRetrieval complete in {total_time:.2f} seconds")
        print(f"Average time per query: {total_time / len(query_ids):.2f} seconds")

        # Build info dictionary with metrics
        infos = {
            "query_embed_time_ms": query_embed_time * 1000,
            "retrieve_time_ms": retrieve_time * 1000,
            "total_search_time_ms": total_time * 1000,
            "retriever_model": self.embedding_model.model_name,
            "modality": self.modality,
            "device": self.embedding_model.device,
            "batch_size": self.batch_size,
            "retriever_top_k": self.top_k
        }

        return results, infos
