import pytest
import torch
from datasets import Dataset
from PIL import Image

from vidore_benchmark.evaluation.vidore_evaluators.vidore_evaluator_beir import BEIRDataset, ViDoReEvaluatorBEIR
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
def mock_beir_dataset() -> BEIRDataset:
    corpus = Dataset.from_dict(
        {
            "corpus-id": [1, 2, 3],
            "image": [Image.new("RGB", (10, 10)) for _ in range(3)],
            "text_description": ["desc1", "desc2", "desc3"],
        }
    )

    queries = Dataset.from_dict(
        {
            "query-id": [1, 2],
            "query": ["query1", "query2"],
        }
    )

    qrels = Dataset.from_dict(
        {
            "query-id": [1, 1, 2],
            "corpus-id": [1, 2, 3],
            "score": [1, 0, 1],
        }
    )

    return {
        "corpus": corpus,
        "queries": queries,
        "qrels": qrels,
    }


@pytest.fixture
def evaluator():
    return ViDoReEvaluatorBEIR(vision_retriever=MockVisionRetriever())


def test_init(evaluator):
    assert isinstance(evaluator.vision_retriever, BaseVisionRetriever)


def test_get_retrieval_results(evaluator):
    query_ids = [1, 2]
    image_ids = [10, 20, 30]
    scores = torch.tensor([[0.8, 0.6, 0.4], [0.7, 0.5, 0.3]])

    results = evaluator._get_retrieval_results(
        query_ids=query_ids,
        image_ids=image_ids,
        scores=scores,
    )

    assert len(results) == 2
    assert "1" in results
    assert "2" in results
    assert len(results["1"]) == 3
    assert results["1"]["10"] == pytest.approx(0.8)
    assert results["1"]["20"] == pytest.approx(0.6)
    assert results["1"]["30"] == pytest.approx(0.4)


def test_evaluate_dataset(evaluator, mock_beir_dataset):
    metrics = evaluator.evaluate_dataset(
        ds=mock_beir_dataset,
        batch_query=2,
        batch_passage=2,
    )

    assert isinstance(metrics, dict)
    assert "ndcg_at_1" in metrics
    assert "map_at_1" in metrics
    assert "recall_at_1" in metrics
    assert "precision_at_1" in metrics
    assert "mrr_at_1" in metrics


def test_evaluate_dataset_with_bm25(mock_bm25_retriever, mock_beir_dataset):
    evaluator = ViDoReEvaluatorBEIR(vision_retriever=mock_bm25_retriever)
    metrics = evaluator.evaluate_dataset(
        ds=mock_beir_dataset,
        batch_query=2,
        batch_passage=2,
    )

    assert isinstance(metrics, dict)
    assert "ndcg_at_1" in metrics
    assert "map_at_1" in metrics
    assert "recall_at_1" in metrics
    assert "precision_at_1" in metrics
    assert "mrr_at_1" in metrics
