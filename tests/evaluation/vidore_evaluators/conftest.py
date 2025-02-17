import pytest
import torch

from vidore_benchmark.retrievers.base_vision_retriever import BaseVisionRetriever
from vidore_benchmark.retrievers.bm25_retriever import BM25Retriever


class MockBM25Retriever(BM25Retriever):
    def __init__(self, device: str = "auto"):
        super().__init__()

    def get_scores_bm25(self, queries, passages, **kwargs):
        return torch.tensor([[0.5 for _ in range(len(passages))] for _ in range(len(queries))])


@pytest.fixture
def mock_bm25_retriever():
    return MockBM25Retriever()


class MockVisionRetriever(BaseVisionRetriever):
    def __init__(self, use_visual_embedding=True):
        self.use_visual_embedding = use_visual_embedding

    def forward_queries(self, queries, batch_size=None, **kwargs):
        return torch.tensor([[1.0, 0.0] for _ in queries])

    def forward_passages(self, passages, batch_size=None, **kwargs):
        return torch.tensor([[0.0, 1.0] for _ in passages])

    def get_scores(self, query_embeddings, passage_embeddings, batch_size=None):
        return torch.tensor([[0.5 for _ in range(len(passage_embeddings))] for _ in range(len(query_embeddings))])

    def get_scores_bm25(self, queries, passages):
        return torch.tensor([[0.5 for _ in range(len(passages))] for _ in range(len(queries))])


@pytest.fixture
def mock_vision_retriever():
    return MockVisionRetriever()
