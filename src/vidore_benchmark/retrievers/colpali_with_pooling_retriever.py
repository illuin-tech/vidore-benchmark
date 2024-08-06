from __future__ import annotations

from typing import List

from dotenv import load_dotenv
from PIL.Image import Image
from torch._tensor import Tensor

from vidore_benchmark.compression.token_pooling import HierarchicalEmbeddingPooler
from vidore_benchmark.retrievers.colpali_retriever import ColPaliRetriever
from vidore_benchmark.retrievers.utils.register_retriever import register_vision_retriever

load_dotenv(override=True)


@register_vision_retriever("vidore/colpali_with_pooling")
class ColPaliWithPoolingRetriever(ColPaliRetriever):
    """
    ColPali Retriever with hierarchical pooling of embeddings.
    """

    def __init__(
        self,
        device: str = "auto",
        pool_factor: int = 2,
    ):
        super().__init__(device=device)
        self.pool_factor = pool_factor
        self.emmbedding_pooler = HierarchicalEmbeddingPooler(pool_factor=self.pool_factor, device=self.device)

    def forward_documents(self, documents: List[Image], batch_size: int, **kwargs) -> List[Tensor]:
        embeddings_doc = super().forward_documents(documents, batch_size, **kwargs)
        embeddings_doc_pooled = [self.emmbedding_pooler.pool_embeddings(embedding) for embedding in embeddings_doc]
        return embeddings_doc_pooled
