#!/usr/bin/env python3
"""
Retrieval visual retrieval pipeline for Vidore v3 Evaluation using NVIDIA's nvidia/llama-nemotron-embed-vl-1b-v2 embedding model + nvidia/llama-nemotron-rerank-vl-1b-v2 reranker model

It demonstrates how to:
1. Subclass BasePipeline for a custom dense retrieval implementation
2. Handle GPU memory constraints by computing embeddings on GPU in batches and storing on CPU
3. Re-rank top-k images retrieved by the embedding model with a reranker model

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
        --module-path example_pipelines/nemotron_embed_and_rerank_vl_v2.py \
        --class-name NemotronEmbedVLPipeline \
        --pipeline-args '{"batch_size": 32, "top_k": 100}'
"""

import sys
import time
from typing import Any, Dict, List
from collections import OrderedDict
from tqdm import tqdm

try:
    import torch
    import torch.nn.functional as F  # noqa: N812
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoProcessor
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

def chunk_list(data, size):
  """Yield successive size-sized chunks from data."""
  return [data[i:i + size] for i in range(0, len(data), size)]

class NemotronEmbedVL():
    """
    Encapsulates logic for the Nemotron Embed VL model
    """    
    def __init__(self, model_name = "nvidia/llama-nemotron-embed-vl-1b-v2", batch_size: int = 32):
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
                device_map=self.device,
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
                batch_embeddings = self.model.encode_documents(images=batch)

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
        # Normalizing embeddings for cosine-similarity scoring
        all_embeddings = _l2_normalize(all_embeddings)
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
                batch_embeddings = self.model.encode_queries(batch)

                # Move to CPU
                batch_embeddings_cpu = batch_embeddings.cpu()
                query_embeddings.append(batch_embeddings_cpu)

                # Clear GPU cache
                del batch_embeddings
                torch.cuda.empty_cache()

        # Concatenate all batches
        all_embeddings = torch.cat(query_embeddings, dim=0)
        # Normalizing embeddings for cosine-similarity scoring
        all_embeddings = _l2_normalize(all_embeddings)
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


class NemotronRerankVL():
    """
    Encapsulates logic for the Nemotron Embed VL model
    """    
    def __init__(self, model_name = "nvidia/llama-nemotron-rerank-vl-1b-v2", batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size

        print(f"Loading {self.model_name}...")
        
        self.device = "cuda"
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            device_map=self.device
        ).eval()

        # Load processor with modality-specific kwargs
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code = True,
            max_input_tiles = 6,
            use_thumbnail = True,
            rerank_max_length = 2048
        )

    def rerank(self, query, query_candidate_corpus_ids, query_candidate_corpus_images):
        examples = [{
            "question": query,
            "doc_text": "",
            "doc_image": image
        } for image in query_candidate_corpus_images]

        batched_examples = chunk_list(examples, self.batch_size)

        rank_scores = []
        for batch in batched_examples:
            # Process with processor
            batch_dict = self.processor.process_queries_documents_crossencoder(batch)
    
            # Move to device
            batch_dict = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch_dict.items()
            }
    
            # Run inference
            with torch.no_grad():
                outputs = self.model(**batch_dict, return_dict=True)

            # Get logits
            logits = outputs.logits
            logits_flat = logits.squeeze(-1)
            rank_scores.extend(logits_flat.cpu().numpy().tolist())

        sorted_candidates = sorted((zip(query_candidate_corpus_ids, rank_scores)), key=lambda x: -x[1])
        sorted_candidates_dict = OrderedDict()
        for corpus_id, score in sorted_candidates:
            sorted_candidates_dict[corpus_id] = score

        return sorted_candidates_dict        
        


class NemotronEmbedRerankVLPipeline(BasePipeline):
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

    def __init__(self, 
                 model_name = "nvidia/llama-nemotron-embed-vl-1b-v2", 
                 batch_size: int = 34, 
                 ranker_batch_size = 1, 
                 top_k: int = 100):
        self.top_k = top_k        
        self.embedding_model = NemotronEmbedVL(model_name="nvidia/llama-nemotron-embed-vl-1b-v2", 
                                               batch_size=batch_size)
        self.reranker = NemotronRerankVL(model_name="nvidia/llama-nemotron-rerank-vl-1b-v2",
                                        batch_size=ranker_batch_size)
        

    def retrieve(
        self,
        query_ids: List[str],
        queries: List[str],
        corpus_ids: List[str],
        corpus_images: List[Any],
        corpus_texts: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        Retrieve relevant corpus items for each query using dense retrieval.

        This method:
        1. Embeds all corpus images on GPU → CPU
        2. Embeds all queries on GPU → CPU
        3. Computes embedding scores on CPU
        4. Re-ranks top-k retrieved results with a reranker model
        4. Returns top-k results per query

        Args:
            query_ids: List of query identifiers
            queries: List of query texts
            corpus_ids: List of corpus item identifiers
            corpus: List of PIL.Image objects

        Returns:
            Dictionary mapping query_id to {corpus_id: score} for top-k results
        """       
        start_time = time.time()

        # Step 1: Embed corpus (GPU → CPU)
        corpus_embeddings = self.embedding_model._embed_corpus_batched(corpus_images)

        # Step 2: Embed queries (GPU → CPU)
        query_embeddings = self.embedding_model._embed_queries_batched(queries)

        # Step 3: Compute scores (CPU)
        scores = self.embedding_model._compute_scores(query_embeddings, corpus_embeddings, batched_scoring=False)

        ######### Retrieving with Embedding #########
        print(f"\nRetrieving top-{self.top_k} candidates per query...")
        results = dict()
        
        # Retrieving top-k corpus items per query using the embedding model
        for q_idx, query_id in enumerate(query_ids):
            # Get scores for this query
            query_scores = scores[q_idx]

            # Get top-k indices and scores
            topk_scores, topk_indices = torch.topk(query_scores, min(self.top_k, len(corpus_ids)))

            # Build results dictionary
            topk_corpus_ids = OrderedDict()
            for idx, score in zip(topk_indices, topk_scores):
                topk_corpus_ids[corpus_ids[idx.item()]] = score.item()
            results[query_id] = topk_corpus_ids

        elapsed = time.time() - start_time
        print(f"\nRetrieval complete in {elapsed:.2f} seconds")
        print(f"Average time per query: {elapsed / len(query_ids):.2f} seconds")             

        ######### Reranking #########
        print(f"\nRerank top-{self.top_k} candidates per query...")
        
        start_time = time.time()
        
        query_id2idx_mapping = dict(zip(query_ids, list(range(len(query_ids)))))
        corpus_id2idx_mapping = dict(zip(corpus_ids, list(range(len(corpus_ids)))))
        
        # Re-ranking top results
        results_reranked = dict()
        for query_id, topk_corpus_ids in results.items():
            query_topk_corpus_ids = []
            query_topk_corpus_images = []
            for corpus_id, corpus_score in topk_corpus_ids.items(): 
                corpus_idx = corpus_id2idx_mapping[corpus_id]
                corpus_image = corpus[corpus_idx]                
                query_topk_corpus_ids.append(corpus_id)
                query_topk_corpus_images.append(corpus_image)

            query_idx = query_id2idx_mapping[query_id]
            query = queries[query_idx]
            results_reranked[query_id] = self.reranker.rerank(query, query_topk_corpus_ids, query_topk_corpus_images)

        elapsed = time.time() - start_time
        print(f"\nReranking complete in {elapsed:.2f} seconds")
        print(f"Average time per query: {elapsed / len(query_ids):.2f} seconds")

        return results_reranked
