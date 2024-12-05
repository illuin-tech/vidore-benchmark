from __future__ import annotations

from typing import List, Optional, Union

import torch
from PIL import Image
from torch import Tensor

from vidore_benchmark.retrievers.registry_utils import register_vision_retriever
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever


@register_vision_retriever("dummy_retriever")
class DummyRetriever(VisionRetriever):
    """
    Dummy retriever that generates random dense embeddings.
    """

    def __init__(
        self,
        emb_dim_query: int = 16,
        emb_dim_doc: int = 16,
        device: str = "cpu",
    ):
        super().__init__()

        self.emb_dim_query = emb_dim_query
        self.emb_dim_doc = emb_dim_doc
        self.device = device

    @property
    def use_visual_embedding(self) -> bool:
        return True

    def forward_queries(self, queries: List[str], batch_size: int, **kwargs) -> Tensor:
        return torch.randn(len(queries), self.emb_dim_query).to(self.device)

    def forward_passages(self, passages: List[Image.Image], batch_size: int, **kwargs) -> Tensor:
        return torch.randn(len(passages), self.emb_dim_doc).to(self.device)

    def get_scores(
        self,
        query_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        passage_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Dot-product similarity between queries and passages.
        """
        if isinstance(query_embeddings, list):
            query_embeddings = torch.stack(query_embeddings)
        if isinstance(passage_embeddings, list):
            passage_embeddings = torch.stack(passage_embeddings)

        scores = torch.einsum("bd,cd->bc", query_embeddings, passage_embeddings)

        return scores
