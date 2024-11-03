from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from datasets import Dataset

from vidore_benchmark.evaluation.eval_utils import CustomRetrievalEvaluator

logger = logging.getLogger(__name__)


class VisionRetriever(ABC):
    """
    Abstract class for vision retrievers used in the ViDoRe benchmark.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        """
        Initialize the VisionRetriever.
        """
        pass

    @property
    @abstractmethod
    def use_visual_embedding(self) -> bool:
        """
        The child class should instantiate the `use_visual_embedding` property:
        - True if the retriever uses native visual embeddings (e.g. JINA-Clip, ColPali)
        - False if the retriever uses text embeddings and possibly VLM-generated captions (e.g. BM25).
        """
        pass

    @abstractmethod
    def forward_queries(
        self,
        queries: Any,
        batch_size: int,
        **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Preprocess and forward pass the queries through the model.

        NOTE: This method can either:
        - return a single tensor where the first dimension corresponds to the number of queries.
        - return a list of tensors where each tensor corresponds to a query.
        """
        pass

    @abstractmethod
    def forward_passages(
        self,
        passages: Any,
        batch_size: int,
        **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Preprocess and forward pass the passages through the model. A passage can a text chunk (e.g. BM25) or
        an image of a document page (e.g. ColPali).

        NOTE: This method can either:
        - return a single tensor where the first dimension corresponds to the number of passages.
        - return a list of tensors where each tensor corresponds to a passage.
        """
        pass

    @abstractmethod
    def get_scores(
        self,
        query_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        passage_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Get the scores between queries and passages.

        Inputs:
        - query_embeddings: torch.Tensor (n_queries, emb_dim_query) or List[torch.Tensor] (emb_dim_query)
        - passage_embeddings: torch.Tensor (n_passages, emb_dim_doc) or List[torch.Tensor] (emb_dim_doc)
        - batch_size: Optional[int]

        Output:
        - scores: torch.Tensor (n_queries, n_passages)
        """
        pass

    def get_relevant_docs_results(
        self,
        ds: Dataset,
        queries: List[str],
        scores: torch.Tensor,
        **kwargs,
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """
        Get the relevant passages and the results from the scores.

        NOTE: Override this method if the retriever has a different output format.

        Outputs:
        - relevant_docs: Dict[str, float]
        {
            "query_0": {"doc_0": 1},
            "query_1": {"doc_1": 1},
            ...
        }
        - results: Dict[str, Dict[str, float]] with shape:
        {
            "query_0": {"doc_i": 19.125, "doc_1": 18.75, ...},
            "query_1": {"doc_j": 17.25, "doc_1": 16.75, ...},
            ...
        }
        """
        relevant_docs = {}
        results = {}

        query_to_filename = {query: image_filename for query, image_filename in zip(ds["query"], ds["image_filename"])}
        passage_to_filename = {docidx: image_filename for docidx, image_filename in enumerate(ds["image_filename"])}

        for query, score_per_query in zip(queries, scores):
            relevant_docs[query] = {query_to_filename[query]: 1}

            for docidx, score in enumerate(score_per_query):
                filename = passage_to_filename[docidx]
                score_passage = float(score.item())

                if query in results:
                    current_score = results[query].get(filename, 0)
                    results[query][filename] = max(current_score, score_passage)
                else:
                    results[query] = {filename: score_passage}

        return relevant_docs, results

    def compute_metrics(
        self,
        relevant_docs: Any,
        results: Any,
        **kwargs,
    ) -> Dict[str, Optional[float]]:
        """
        Compute the MTEB metrics.

        NOTE: Override this method if the retriever has a different evaluation metric.
        """

        mteb_evaluator = CustomRetrievalEvaluator()

        ndcg, _map, recall, precision, naucs = mteb_evaluator.evaluate(
            relevant_docs,
            results,
            mteb_evaluator.k_values,
            ignore_identical_ids=kwargs.get("ignore_identical_ids", True),
        )

        mrr = mteb_evaluator.evaluate_custom(relevant_docs, results, mteb_evaluator.k_values, "mrr")

        scores = {
            **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
            **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
            **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
            **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
            **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr[0].items()},
            **{f"naucs_at_{k.split('@')[1]}": v for (k, v) in naucs.items()},
        }

        return scores
