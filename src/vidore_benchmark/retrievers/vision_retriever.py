from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

import torch

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
