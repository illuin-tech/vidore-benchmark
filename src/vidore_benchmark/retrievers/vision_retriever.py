from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, cast

import torch
from datasets import Dataset
from PIL import Image
from torch.utils.data import DataLoader

from vidore_benchmark.evaluation.retrieval_evaluator import CustomEvaluator


class VisionRetriever(ABC):
    """
    Abstract class for ViDoRe retrievers.
    """

    text_only: bool  # TODO : change name
    is_multi_vector: bool

    def __init__(
        self,
        *args,
        **kwargs,
    ):
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
    def get_scores(self, 
                    queries : List[str],
                    documents : List[Image.Image | str], 
                    batch_query : int, 
                    batch_doc : int) -> torch.Tensor:
        """
        Get the scores for the documents and queries.
        """
        pass
    
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

def evaluate_dataset(
    vision_retriever: VisionRetriever,
    ds: Dataset,
    batch_query: int,
    batch_doc: int,
) -> Dict[str, float]:
    """
    Evaluate the model on a given dataset using the MTEB metrics.
    """

    # Dataset: sanity check
    col_documents = "text_description" if vision_retriever.text_only else "image"
    col_to_check = ["query", col_documents, "image_filename"]

    if not all(col in ds.column_names for col in col_to_check):
        raise ValueError(f"Dataset should contain the following columns: {col_to_check}")
    
    # Remove None queries and duplicates
    queries = list(set(ds["query"]))
    if None in queries:
        queries.remove(None)
        if len(queries) == 0:
            raise ValueError("All queries are None")

    documents = ds[col_documents]

    # Get the scores - size (n_queries, n_documents)
    scores = vision_retriever.get_scores(queries, documents, batch_query=batch_query, batch_doc=batch_doc)


    # Get the relevant documents and results
    relevant_docs, results = vision_retriever.get_relevant_docs_results(ds, queries, scores)

    evaluator = CustomEvaluator(is_multi_vector=False)  # TODO Change

    # compute MTEB metrics

    metrics = evaluator.compute_metrics(relevant_docs, results)

    return metrics


def get_top_k(
        vision_retriever : VisionRetriever,
        queries: List[str],
        ds: Dataset,
        batch_size: int,
        k: int,
    ) -> Dict[str, float]:
        """
        Get the top-k documents for a given query.
        """

        raise NotImplementedError("Implement the logic to get the top-k documents")


