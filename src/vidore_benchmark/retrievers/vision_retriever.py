from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, cast

import torch
from datasets import Dataset
from PIL import Image
from torch.utils.data import DataLoader


from mteb.evaluation.evaluators import RetrievalEvaluator


class VisionRetriever(ABC):
    """
    Abstract class for ViDoRe retrievers.

    Args:
    - visual_embedding: bool (whether the retriever uses visual embeddings or not)
    """

    visual_embedding: bool 

    def __init__(
        self,
        visual_embedding: bool,
        *args,
        **kwargs,
    ):
        self.visual_embedding = visual_embedding
        pass

    @abstractmethod
    def forward_queries(self, queries, **kwargs) -> torch.Tensor:
        """
        Forward pass the processed queries.
        """
        pass

    @abstractmethod
    def forward_documents(self, documents, **kwargs) -> torch.Tensor:
        """
        Forward pass the processed documents (i.e. page images).
        """
        pass

    @abstractmethod
    def get_scores(
        self, queries: List[str], documents: List[Image.Image | str], batch_query: int, batch_doc: int
    ) -> torch.Tensor:
        """
        Get the scores for the documents and queries.
        """
        pass

    # Can be overwritten if needed
    def get_relevant_docs_results(
        self, ds: Dataset, queries: List[str], scores: torch.Tensor
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """
        Get the relevant documents and the results from the scores.

        Inputs:
        - queries: List[str]
        - documents: List[str]
        - scores: torch.Tensor (n_queries, n_documents)

        Outputs:
        - relevant_docs: Dict[str, float] (document -> 1) for each query (i.e. only one relevant document per query)
        - results: Dict[str, Dict[str, float]] (query -> {document: score}) for each query
        """

        relevant_docs = {}
        results = {}

        queries2filename = {query: image_filename for query, image_filename in zip(ds["query"], ds["image_filename"])}
        passages2filename = {docidx: image_filename for docidx, image_filename in enumerate(ds["image_filename"])}

        for query, score_per_query in zip(queries, scores):
            relevant_docs[query] = {queries2filename[query]: 1}

            for docidx, score in enumerate(score_per_query):
                filename = passages2filename[docidx]
                score_passage = float(score.item())

                if query in results:
                    results[query][filename] = max(results[query].get(filename, 0), score_passage)
                else:
                    results[query] = {filename: score_passage}

        return relevant_docs, results

    # Can be overwritten if needed
    def compute_metrics(self, relevant_docs, results, **kwargs):
        mteb_evaluator = RetrievalEvaluator()
        ndcg, _map, recall, precision, naucs = mteb_evaluator.evaluate(  # type: ignore
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

def get_top_k(
    vision_retriever: VisionRetriever,
    queries: List[str],
    ds: Dataset,
    batch_size: int,
    k: int,
) -> Dict[str, float]:
    """
    Get the top-k documents for a given query.
    """

    raise NotImplementedError("Implement the logic to get the top-k documents")
