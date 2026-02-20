"""
Base class for implementing pipelines.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union


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
        corpus_images: List[Any],
        corpus_texts: List[str],
    ) -> Union[Dict[str, Dict[str, float]], Tuple[Dict[str, Dict[str, float]], Optional[Dict[str, Any]]]]:
        """
        Retrieve relevant corpus items for each query.

        Args:
            query_ids: List of query identifiers (e.g., ['q1', 'q2', 'q3'])
            queries: List of query texts corresponding to query_ids
            corpus_ids: List of corpus item identifiers (e.g., ['doc1', 'doc2', ...])
            corpus_images: List of corpus images (PIL.Image objects in vidore v3)
            corpus_texts: List of corpus texts (markdown strings in vidore v3)

        Returns:
            Either:
            - A dictionary mapping query_id to a dictionary of corpus_id: score pairs.
            - A tuple of (results_dict, infos_dict) where infos_dict contains optional
              tracking metrics such as cost, granular timing, num_gpus, etc.

            Scores should be floats where higher values indicate higher relevance.
            The infos dictionary is optional and can be None or omitted entirely.

            Example return format:
            {
                'q1': {
                    'doc1': 0.95,
                    'doc3': 0.87,
                    'doc5': 0.72,
                    ...
                },
                'q2': {
                    'doc2': 0.91,
                    'doc1': 0.83,
                    ...
                },
                ...
            }

        Note:
            - You don't need to return scores for all corpus items, only the top-k
            - The returned dictionary will be automatically converted to pytrec_eval format
            - Scores are relative; only the ranking matters for NDCG@10
        """
        pass
