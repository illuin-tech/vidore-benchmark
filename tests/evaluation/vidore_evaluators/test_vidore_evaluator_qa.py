import pytest
import torch
from datasets import Dataset
from PIL import Image

from vidore_benchmark.evaluation.vidore_evaluators.vidore_evaluator_qa import ViDoReEvaluatorQA
from vidore_benchmark.retrievers.base_vision_retriever import BaseVisionRetriever


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
def mock_dataset():
    return Dataset.from_dict(
        {
            "query": ["query1", "query2", "query1", None],
            "image": [Image.new("RGB", (10, 10)) for _ in range(4)],
            "text_description": ["desc1", "desc2", "desc3", "desc4"],
            "image_filename": ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"],
        }
    )


@pytest.fixture
def evaluator():
    return ViDoReEvaluatorQA(vision_retriever=MockVisionRetriever())


def test_init(evaluator):
    assert isinstance(evaluator.vision_retriever, BaseVisionRetriever)


def test_evaluate_dataset(evaluator, mock_dataset):
    metrics = evaluator.evaluate_dataset(
        ds=mock_dataset,
        batch_query=2,
        batch_passage=2,
    )

    assert isinstance(metrics, dict)
    assert "ndcg_at_1" in metrics
    assert "map_at_1" in metrics
    assert "recall_at_1" in metrics
    assert "precision_at_1" in metrics
    assert "mrr_at_1" in metrics


def test_evaluate_dataset_with_bm25(mock_bm25_retriever, mock_dataset):
    evaluator = ViDoReEvaluatorQA(vision_retriever=mock_bm25_retriever)
    metrics = evaluator.evaluate_dataset(
        ds=mock_dataset,
        batch_query=2,
        batch_passage=2,
    )

    assert isinstance(metrics, dict)
    assert "ndcg_at_1" in metrics
    assert "map_at_1" in metrics
    assert "recall_at_1" in metrics
    assert "precision_at_1" in metrics
