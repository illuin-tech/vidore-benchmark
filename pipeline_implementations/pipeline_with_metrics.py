"""
Example pipeline that returns additional tracking metrics.

This example demonstrates how to return optional tracking information
alongside retrieval results, such as cost, timing, GPU usage, etc.
"""

import random
import time
from typing import Any, Dict, List

from vidore_benchmark.pipeline_evaluation.base_pipeline import BasePipeline


class PipelineWithMetrics(BasePipeline):
    """
    An example pipeline that tracks and returns additional metrics.

    This demonstrates the optional second return value feature, allowing
    pipelines to report metrics like:
    - Computational cost (e.g., API costs, GPU hours)
    - Granular timing information
    - Resource usage (GPU count, memory, etc.)
    - Model-specific metadata

    Args:
        top_k: Number of results to return per query (default: 10)
        track_metrics: Whether to return additional metrics (default: True)
    """

    def __init__(self, top_k: int = 10, track_metrics: bool = True):
        self.top_k = top_k
        self.track_metrics = track_metrics
        self.rng = random.Random(42)

    def index(
        self,
        corpus_ids: List[str],
        corpus_images: List[Any],
        corpus_texts: List[Any],
    ) -> None:
        """
        Index the corpus by storing corpus IDs and simulating encoding.

        Args:
            corpus_ids: List of corpus item identifiers
            corpus_images: List of corpus images
            corpus_texts: List of corpus texts
        """
        self.corpus_ids = corpus_ids

        # Simulate corpus encoding
        encode_start = time.time()
        time.sleep(0.01)  # Simulate processing
        self.encode_time = time.time() - encode_start

    def search(
        self,
        query_ids: List[str],
        queries: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        Search the indexed corpus and optionally track metrics.

        Args:
            query_ids: List of query identifiers
            queries: List of query texts

        Returns:
            A dictionary mapping query_id to {corpus_id: score} pairs.
            If track_metrics is True, also returns an infos dictionary with metrics.
            - results: Dictionary mapping query_id to {corpus_id: score} pairs
            - infos: Optional dictionary with tracking metrics (can be None)
        """
        # Track timing for search phase
        start_time = time.time()

        # Perform retrieval (simplified random retrieval for demo)
        results = {}
        for query_id in query_ids:
            k = min(self.top_k, len(self.corpus_ids))
            sampled_corpus_ids = self.rng.sample(self.corpus_ids, k)
            query_results = {corpus_id: self.rng.random() for corpus_id in sampled_corpus_ids}
            results[query_id] = query_results
        retrieval_time = time.time() - start_time

        total_time = self.encode_time + retrieval_time

        # Optionally return additional metrics
        if self.track_metrics:
            infos = {
                # Timing breakdown
                "encode_time_ms": self.encode_time * 1000,
                "retrieval_time_ms": retrieval_time * 1000,
                "total_time_ms": total_time * 1000,
                # Resource usage
                "num_gpus": 1,
                "gpu_memory_gb": 16.0,
                # Cost tracking (example for API-based models)
                "estimated_cost_usd": 0.05,
                "num_api_calls": len(queries) + len(self.corpus_ids),
                # Model metadata
                "model_name": "example-model-v1.0",
                "embedding_dim": 768,
                # Retrieval statistics
                "num_queries": len(query_ids),
                "corpus_size": len(self.corpus_ids),
            }
            return results, infos
        else:
            return results
