"""
Base class for implementing pipelines.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BasePipeline(ABC):
    """
    Abstract base class for pipelines.

    Users should inherit from this class and implement the retrieve() method
    with their custom pipeline logic.
    """

    @abstractmethod
    def retrieve(
        self,
        query_ids: List[str],
        queries: List[str],
        corpus_ids: List[str],
        corpus: List[Any],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve relevant corpus items for each query.

        Args:
            query_ids: List of query identifiers (e.g., ['q1', 'q2', 'q3'])
            queries: List of query texts corresponding to query_ids
            corpus_ids: List of corpus item identifiers (e.g., ['doc1', 'doc2', ...])
            corpus: List of corpus items (images as PIL.Image objects in vidore v3)

        Returns:
            Dictionary mapping query_id to a dictionary with:
              - "results": Dict[str, float] of corpus_id -> score pairs
              - "runtime_milliseconds": float runtime for that query in milliseconds

            Scores should be floats where higher values indicate higher relevance.

            Example return format:
            {
                "q1": {
                    "results": {
                        "doc1": 0.95,
                        "doc3": 0.87,
                        "doc5": 0.72,
                        ...
                    },
                    "runtime_milliseconds": 41.2,
                },
                "q2": {
                    "results": {
                        "doc2": 0.91,
                        "doc1": 0.83,
                        ...
                    },
                    "runtime_milliseconds": 38.8,
                },
                ...
            }

        Note:
            - You don't need to return scores for all corpus items, only the top-k
            - The returned dictionary will be converted to pytrec_eval format internally
            - Scores are relative; only the ranking matters for NDCG@10
        """
        pass
