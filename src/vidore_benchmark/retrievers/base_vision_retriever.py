from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

import torch

logger = logging.getLogger(__name__)


class BaseVisionRetriever(ABC):
    """
    Base class for vision retrievers used in the ViDoRe benchmark.
    """

    def __init__(self, use_visual_embedding: bool, **kwargs):
        """
        Initialize the VisionRetriever.

        Args:
            use_visual_embedding (bool): Whether the retriever uses visual embeddings.
                Set to:
                - True if the retriever uses native visual embeddings (e.g. JINA-Clip, ColPali)
                - False if the retriever uses text embeddings and possibly VLM-generated captions (e.g. BM25).
        """
        self.use_visual_embedding = use_visual_embedding

    @abstractmethod
    def forward_queries(
        self,
        queries: Any,
        batch_size: int,
        **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Preprocess and forward pass the queries through the model.

        Args:
            queries (Any): The queries to forward pass.
            batch_size (int): The batch size for the queries
            **kwargs: Additional keyword arguments.

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: The query embeddings.
                This can either be:
                - a single tensor where the first dimension corresponds to the number of queries.
                - a list of tensors where each tensor corresponds to a query.
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

        Args:
            passages (Any): The passages to forward pass.
            batch_size (int): The batch size for the passages.
            **kwargs: Additional keyword arguments.

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: The passage embeddings.
                This can either be:
                - a single tensor where the first dimension corresponds to the number of passages.
                - a list of tensors where each tensor corresponds to a passage.
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

        Args:
            query_embeddings (Union[torch.Tensor, List[torch.Tensor]]): The query embeddings.
                This can either be:
                - a single tensor where the first dimension corresponds to the number of queries.
                - a list of tensors where each tensor corresponds to a query.
            passage_embeddings (Union[torch.Tensor, List[torch.Tensor]]): The passage embeddings.
                This can either be:
                - a single tensor where the first dimension corresponds to the number of passages.
                - a list of tensors where each tensor corresponds to a passage.
            batch_size (Optional[int]): The batch size for the scoring.

        Returns:
            torch.Tensor: The similarity scores between queries and passages. Shape: (n_queries, n_passages).
        """
        pass
