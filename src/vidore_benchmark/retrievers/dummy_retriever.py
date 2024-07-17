from __future__ import annotations

import math
from typing import List, Optional

import torch
from PIL import Image
from torch import Tensor

from vidore_benchmark.retrievers.utils.register_retriever import register_vision_retriever
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever


@register_vision_retriever("dummy_retriever")
class DummyRetriever(VisionRetriever):
    """
    Dummy retriever for testing purposes. It generates random embeddings and scores.

    NOTE: The dummy retriever takes PIL images in its `forward_documents` method.
    """

    def __init__(
        self,
        emb_dim_query: int = 512,
        emb_dim_doc: int = 512,
    ):
        super().__init__()
        self.emb_dim_query = emb_dim_query
        self.emb_dim_doc = emb_dim_doc

    @property
    def use_visual_embedding(self) -> bool:
        return True

    def forward_queries(self, queries: List[str], batch_size: int, **kwargs) -> List[Tensor]:
        return [torch.randn(batch_size, self.emb_dim_query) for _ in range(math.ceil(len(queries) / batch_size))]

    def forward_documents(self, documents: List[Image.Image], batch_size: int, **kwargs) -> List[Tensor]:
        return [torch.randn(batch_size, self.emb_dim_doc) for _ in range(math.ceil(len(documents) / batch_size))]

    def get_scores(
        self,
        list_emb_queries: List[torch.Tensor],
        list_emb_documents: List[torch.Tensor],
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        return torch.randint(0, 1, (len(list_emb_queries), len(list_emb_documents)))
