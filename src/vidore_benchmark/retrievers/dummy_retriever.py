from typing import List

import torch
from PIL import Image

from vidore_benchmark.retrievers.utils.register_models import register_vision_retriever
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever


@register_vision_retriever("dummy_retriever")
class DummyRetriever(VisionRetriever):
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
        return False

    def forward_queries(self, queries: List[str], **kwargs) -> torch.Tensor:
        return torch.randn(len(queries), self.emb_dim_query)

    def forward_documents(self, documents: List[Image.Image], **kwargs) -> torch.Tensor:
        return torch.randn(len(documents), self.emb_dim_doc)

    def get_scores(
        self,
        queries: List[str],
        documents: List["Image.Image | str"],
        batch_query: int,
        batch_doc: int,
        **kwargs,
    ) -> torch.Tensor:
        scores = torch.randn(len(queries), len(documents))
        return scores
