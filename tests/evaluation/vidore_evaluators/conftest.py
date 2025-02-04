import pytest
import torch

from vidore_benchmark.retrievers.bm25_retriever import BM25Retriever


class MockBM25Retriever(BM25Retriever):
    def __init__(self, device: str = "auto"):
        super().__init__()

    def get_scores_bm25(self, queries, passages, **kwargs):
        return torch.tensor([[0.5 for _ in range(len(passages))] for _ in range(len(queries))])


@pytest.fixture
def mock_bm25_retriever():
    return MockBM25Retriever()
