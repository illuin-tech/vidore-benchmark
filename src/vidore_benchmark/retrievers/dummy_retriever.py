from __future__ import annotations

from typing import List, Tuple

import torch
from PIL import Image

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

    def forward_queries(self, queries: List[str], **kwargs) -> torch.Tensor:
        return torch.randn(len(queries), self.emb_dim_query)

    def forward_documents(self, documents: List[Image.Image], **kwargs) -> torch.Tensor:
        return torch.randn(len(documents), self.emb_dim_doc)

    def get_embeddings(
        self,
        queries: List[str],
        documents: List[Image.Image] | List[str],
        batch_query: int,
        batch_doc: int,
        **kwargs,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

        return (
            [torch.randn(batch_query, self.emb_dim_query) for _ in range(len(queries) // batch_query)],
            [torch.randn(batch_doc, self.emb_dim_doc) for _ in range(len(documents) // batch_doc)],
        )

    def get_scores(
        self,
        list_emb_queries: List[torch.Tensor],
        list_emb_documents: List[torch.Tensor],
    ) -> torch.Tensor:
        return torch.randn(len(list_emb_queries), len(list_emb_documents))
