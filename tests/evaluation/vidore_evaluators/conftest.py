import pytest
import torch

from vidore_benchmark.retrievers.base_vision_retriever import BaseVisionRetriever


class MockBM25Retriever(BaseVisionRetriever):
    def __init__(self, device: str = "auto"):
        super().__init__(use_visual_embedding=False)

    def forward_queries(self, queries, batch_size=None, **kwargs):
        raise NotImplementedError("BM25Retriever only need get_scores_bm25 method.")

    def forward_passages(self, passages, batch_size=None, **kwargs):
        raise NotImplementedError("BM25Retriever only need get_scores_bm25 method.")

    def get_scores(self, query_embeddings, passage_embeddings, batch_size=None):
        raise NotImplementedError("Please use the `get_scores_bm25` method instead.")

    def get_scores_bm25(self, queries, passages, **kwargs):
        return torch.tensor([[0.5 for _ in range(len(passages))] for _ in range(len(queries))])


@pytest.fixture
def mock_bm25_retriever():
    return MockBM25Retriever()
