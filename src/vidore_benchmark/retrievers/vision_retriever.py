from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import torch
from datasets import Dataset
from mteb.evaluation.evaluators import RetrievalEvaluator
from PIL import Image


class VisionRetriever(ABC):
    """
    Abstract class for ViDoRe retrievers.
    """

    def __init__(self):
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
    def forward_queries(self, queries: Any, **kwargs) -> torch.Tensor | List[torch.Tensor]:
        """
        Forward pass the processed queries.

        NOTE: This method can either:
        - return a single tensor where the first dimension corresponds to the number of queries.
        - return a list of tensors where each tensor corresponds to a query.
        """
        pass

    @abstractmethod
    def forward_documents(self, documents: Any, **kwargs) -> torch.Tensor | List[torch.Tensor]:
        """
        Forward pass the processed documents (i.e. page images).

        NOTE: This method can either:
        - return a single tensor where the first dimension corresponds to the number of documents.
        - return a list of tensors where each tensor corresponds to a document.
        """
        pass

    @abstractmethod
    def get_scores(
        self,
        queries: List[str],
        documents: List[Image.Image] | List[str],
        batch_query: int,
        batch_doc: int,
        **kwargs,
    ) -> torch.Tensor:
        """
        Get the similarity scores between queries and documents.

        `documents` can be a list of:
        - PIL images olist of image filenames
        - filepaths (strings) of the images
        - OCR-ed text (strings) of the images.
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
        Get the relevant documents and the results from the scores.

        NOTE: Override this method if the retriever has a different output format.

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

    def compute_metrics(self, relevant_docs, results, **kwargs):
        """
        Compute the MTEB metrics.

        NOTE: Override this method if the retriever has a different evaluation metric.
        """
        mteb_evaluator = RetrievalEvaluator()
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
