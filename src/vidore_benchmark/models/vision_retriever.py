from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, cast

import torch
from datasets import Dataset
from PIL import Image
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, ProcessorMixin

from vidore_benchmark.dataset.vision_collator import VisionCollator
from vidore_benchmark.evaluation.retrieval_evaluator import CustomEvaluator


class VisionRetriever(ABC):
    """
    Abstract class for ViDoRe retrievers.
    """

    model: torch.nn.Module | PreTrainedModel
    is_multi_vector: bool
    processor: ProcessorMixin | None
    collator: VisionCollator | None

    def __init__(
        self,
        model,
        processor,
        collator,
        *args,
        **kwargs,
    ):
        pass

    @abstractmethod
    def to(self, device: str | torch.device) -> VisionRetriever:
        """
        Send the model and all the necessary components to the specified device.
        """
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
    def embed_queries(self, queries: List[str]) -> torch.Tensor:
        """
        Embed a list of queries into a batched tensor.
        """
        # TODO: processor -> self.forward_queries -> return embeddings
        pass

    @abstractmethod
    def embed_documents(self, documents: List[Image.Image]) -> torch.Tensor:
        """
        Embed a list of documents (i.e. a list of page images) into a batched tensor.
        """
        # TODO: processor -> self.forward_documents -> return embeddings
        pass

    def get_relevant_docs_results(
        self, queries: List[str], documents: List[str], scores: torch.Tensor
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

        pass

    def evaluate_dataset(
        self,
        ds: Dataset,
        batch_size: int,
    ) -> Dict[str, float]:
        """
        Evaluate the model on a given dataset using the MTEB metrics.
        """
        # Dataset: sanity check
        # TODO: assert if all the necessary columns are present in the dataset

        # Create the dataloader
        dataloader = DataLoader(ds, batch_size=batch_size, collate_fn=self.collator)

        # Create the evaluator
        evaluator = CustomEvaluator(is_multi_vector=self.is_multi_vector)

        # Placeholder for the embeddings
        list_emb_queries: List[torch.Tensor] = []
        list_emb_documents: List[torch.Tensor] = []

        for batch in dataloader:
            batch = cast(Dict[str, torch.Tensor], batch)
            # Embed the queries and documents
            # NOTE: in the original code: `emb_queries` -> `qs`, `emb_documents` -> `ps`
            if self.collator is None:
                # TODO: what if we don't use a collator?
                pass
            else:
                list_emb_queries.append(self.forward_queries(batch[self.collator.col_query]))
                list_emb_documents.append(self.forward_documents(batch[self.collator.col_document]))

        # Concatenate the embeddings
        emb_queries = torch.cat(list_emb_queries, dim=0)
        emb_documents = torch.cat(list_emb_documents, dim=0)

        # Evaluate the model
        scores = evaluator.get_scores_matrix(emb_queries, emb_documents)

        assert scores.shape == (
            len(ds[self.collator.col_query]),
            len(ds[self.collator.col_document]),
        ), f"Scores shape is {scores.shape} instead of {(len(ds[self.collator.col_query]), len(ds[self.collator.col_document]))}"

        # Compute the metrics
        relevant_docs, results = self.get_relevant_docs_results(
            ds[self.collator.col_query], ds[self.collator.col_document], scores
        )

        metrics = evaluator.compute_metrics(relevant_docs, results)

        # TODO ? : return also the times ?
        return metrics

    def get_top_k(
        self,
        queries: List[str],
        ds: Dataset,
        batch_size: int,
        k: int,
    ) -> Dict[str, float]:
        """
        Get the top-k documents for a given query.
        """
        # Dataset: sanity check
        # TODO: assert if all the necessary columns are present in the dataset

        # Create the dataloader
        dataloader = DataLoader(ds, batch_size=batch_size, collate_fn=self.collator)

        evaluator = CustomEvaluator(is_multi_vector=self.is_multi_vector)

        # Placeholder for the embeddings
        # TODO: what if we don't use a collator?
        list_emb_queries: List[torch.Tensor] = []
        for query in queries:
            list_emb_queries.append(self.forward_queries(batch[self.collator.col_query]))

        list_emb_documents: List[torch.Tensor] = []
        for batch in dataloader:
            batch = cast(Dict[str, torch.Tensor], batch)
            list_emb_documents.append(self.forward_documents(batch[self.collator.col_document]))

        # Concatenate the embeddings
        emb_query = torch.cat(list_emb_queries, dim=0)
        emb_documents = torch.cat(list_emb_documents, dim=0)

        # Evaluate the model

        scores = evaluator.get_scores_matrix(emb_query, emb_documents)

        assert scores.shape == (
            len(queries),
            len(ds[self.collator.col_document]),
        ), f"Scores shape is {scores.shape} instead of {(len(queries), len(ds[self.collator.col_document]))}"

        # Get the top-k documents
        # TODO: implement logic here

        relevant_docs, results = self.get_relevant_docs_results(queries, ds[self.collator.col_document], scores)

        # sort the results
        top_k: Dict[str, float] = {}

        for query in queries:
            top_k[query] = sorted(results[query].items(), key=lambda x: x[1], reverse=True)[:k]

        return top_k
