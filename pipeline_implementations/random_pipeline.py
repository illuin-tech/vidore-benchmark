"""
Random baseline pipeline for testing and benchmarking.
"""

import random
from typing import Dict, List

from vidore_benchmark.pipeline_evaluation.base_pipeline import BasePipeline


class RandomPipeline(BasePipeline):
    """
    A baseline pipeline that returns random scores.

    Useful for:
    - Testing the evaluation pipeline
    - Providing a baseline for comparison
    - Demonstrating how to implement BasePipeline

    Args:
        seed: Random seed for reproducibility (default: 42)
        top_k: Number of results to return per query (default: 10)
    """

    def __init__(self, seed: int = 42, top_k: int = 10):
        self.seed = seed
        self.top_k = top_k
        self.rng = random.Random(seed)

    def search(
        self,
        query_ids: List[str],
        queries: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        Retrieve random corpus items for each query.

        Returns random scores for a random subset of corpus items.
        """
        results = {}

        for query_id in query_ids:
            # Randomly sample top_k corpus items (or all if corpus is smaller)
            k = min(self.top_k, len(self.corpus_ids))
            sampled_corpus_ids = self.rng.sample(self.corpus_ids, k)

            # Assign random scores to sampled items
            query_results = {corpus_id: self.rng.random() for corpus_id in sampled_corpus_ids}

            results[query_id] = query_results

        additional_info = {
            "seed": self.seed,
            "top_k": self.top_k,
        }

        return results, additional_info
