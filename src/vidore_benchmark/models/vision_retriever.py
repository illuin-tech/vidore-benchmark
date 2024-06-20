from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, cast

import torch
from datasets import Dataset
from FlagEmbedding import BGEM3FlagModel
from PIL import Image
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, ProcessorMixin
from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm

from vidore_benchmark.dataset.vision_collator import VisionCollator
from vidore_benchmark.evaluation.retrieval_evaluator import CustomEvaluator


class VisionRetriever(ABC):
    """
    Abstract class for ViDoRe retrievers.
    """

    model: torch.nn.Module | PreTrainedModel | BGEM3FlagModel
    is_vision_retriever: bool
    is_multi_vector: bool
    processor: ProcessorMixin | None
    collator: VisionCollator

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
    def embed_documents(self, documents: List[Image.Image | str]) -> torch.Tensor:
        """
        Embed a list of documents (i.e. a list of page images) into a batched tensor.
        """
        # TODO: processor -> self.forward_documents -> return embeddings
        pass

    def get_relevant_docs_results(
        self, ds : Dataset, queries : List[str], scores: torch.Tensor
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
        self,
        ds: Dataset,
        batch_size: int,
    ) -> Dict[str, float]:
        """
        Evaluate the model on a given dataset using the MTEB metrics.
        """
        # Dataset: sanity check
        if self.is_vision_retriever:
            # Vision retriever: check if the necessary columns are present
            col_to_check = ['image', 'query', 'image_filename']
            if not all([col in ds.column_names for col in col_to_check]):
                raise ValueError(f"Dataset should contain the following columns: {col_to_check}")
        else:
            # Text retriever: check if the necessary columns are present
            col_to_check = ['query', 'text_description', 'image_filename']
            if not all([col in ds.column_names for col in col_to_check]):
                raise ValueError(f"Dataset should contain the following columns: {col_to_check}")

        # Create the dataloader
        dataloader = DataLoader(ds, batch_size=batch_size, collate_fn=self.collator)

        # Create the evaluator
        evaluator = CustomEvaluator(is_multi_vector=self.is_multi_vector)

        # Placeholder for the embeddings
        list_emb_queries: List[torch.Tensor] = []
        list_emb_documents: List[torch.Tensor] = []

        print("Embedding queries")
        # Embed queries with batch size 1

        queries = list(set(ds[self.collator.col_query]))
        if None in queries:
            queries.remove(None)
        for query in tqdm(queries):
            list_emb_queries.append(self.forward_queries(query))

        print("Embedding documents")
        for batch in tqdm(dataloader):
            # Embed documents with batch size
            # NOTE: in the original code: `emb_queries` -> `qs`, `emb_documents` -> `ps`
            list_emb_documents.append(self.forward_documents(batch['document']))

        # Concatenate the embeddings

        emb_queries = torch.stack(list_emb_queries, dim=0)      # (n_queries, emb_dim)
        emb_documents = torch.cat(list_emb_documents, dim=0)    # (n_documents, emb_dim)

        # Evaluate the model
        scores = evaluator.get_scores_matrix(emb_queries, emb_documents) # (n_queries, n_documents)

        assert scores.shape == (
            len(queries),
            len(ds[self.collator.col_document]),
        ), f"Scores shape is {scores.shape} instead of {(len(queries), len(ds[self.collator.col_document]))}"

        # Compute the metrics
        relevant_docs, results = self.get_relevant_docs_results(ds, queries, scores)

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
            list_emb_queries.append(self.forward_queries(query))

        list_emb_documents: List[torch.Tensor] = []
        for batch in dataloader:
            batch = cast(Dict[str, torch.Tensor], batch)
            list_emb_documents.append(self.forward_documents(batch["document"]))

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

        relevant_docs, results = self.get_relevant_docs_results(queries, ds['image_filename'], scores)

        # sort the results
        top_k: Dict[str, float] = {}

        raise NotImplementedError("Implement the logic to get the top-k documents")
