from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, cast

import torch
from datasets import Dataset
from mteb.evaluation.evaluators import RetrievalEvaluator
from PIL import Image
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, ProcessorMixin

from vidore_benchmark.dataset.vision_collator import VisionCollator


class VisionRetriever(ABC):
    """
    Abstract class for ViDoRe retrievers.
    """

    model: torch.nn.Module | PreTrainedModel
    processor: ProcessorMixin
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
    def forward_queries(self, **kwargs) -> torch.Tensor:
        """
        Forward pass the processed queries.
        """
        pass

    @abstractmethod
    def forward_documents(self, **kwargs) -> torch.Tensor:
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

    @abstractmethod
    def get_similarity_matrix(
        self,
        emb_queries: torch.Tensor,
        emb_documents: torch.Tensor,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute the similarity matrix between a batch of queries and a batch of documents.

        Inputs:
        - emb_queries: (n_queries, emb_dim)
        - emb_documents: (n_documents, emb_dim)
        - batch_size: int (optional), some models require batching as the similarity matrix computation is memory-intensive

        Output:
        - similarity_matrix: (n_queries, n_documents)
        """
        # TODO: for ColPali, import and use the ColBERT scoring method (use einsum for efficiency)
        pass


def compute_mteb_metrics(relevant_docs, results, **kwargs) -> Dict[str, float]:
    """
    TODO: Fill in the docstring.
    TODO: type the arg inputs
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


def evaluate(
    retriever: VisionRetriever,
    ds: Dataset,
    batch_size: int,
) -> Dict[str, float]:
    """
    Evaluate the model on a given dataset using the MTEB metrics.
    """
    # Dataset: sanity check
    # TODO: assert if all the necessary columns are present in the dataset

    # Create the dataloader
    dataloader = DataLoader(ds, batch_size=batch_size, collate_fn=retriever.collator)

    # Placeholder for the embeddings
    list_emb_queries: List[torch.Tensor] = []
    list_emb_documents: List[torch.Tensor] = []

    for batch in dataloader:
        batch = cast(Dict[str, torch.Tensor], batch)
        # Embed the queries and documents
        # NOTE: in the original code: `emb_queries` -> `qs`, `emb_documents` -> `ps`
        list_emb_queries.append(retriever.forward_queries(batch[retriever.collator.col_query]))
        list_emb_documents.append(retriever.forward_documents(batch[retriever.collator.col_query]))

    # Concatenate the embeddings
    emb_queries = torch.cat(list_emb_queries, dim=0)
    emb_documents = torch.cat(list_emb_documents, dim=0)

    # Compute the similarity matrix
    similarity_matrix = retriever.get_similarity_matrix(emb_queries, emb_documents)  # (n_queries, n_documents)

    # Compute the metrics
    metrics = compute_mteb_metrics(similarity_matrix)  # FIXME

    return metrics
