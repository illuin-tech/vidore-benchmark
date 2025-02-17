import pytest
from datasets import Dataset
from PIL import Image

from vidore_benchmark.evaluation.vidore_evaluators.vidore_evaluator_beir import BEIRDataset, ViDoReEvaluatorBEIR
from vidore_benchmark.retrievers.base_vision_retriever import BaseVisionRetriever


@pytest.fixture
def mock_beir_dataset() -> BEIRDataset:
    corpus = Dataset.from_dict(
        {
            "corpus-id": [1, 2, 3, 4],
            "image": [Image.new("RGB", (10, 10)) for _ in range(4)],
            "text_description": ["desc1", "desc2", "desc3", "desc4"],
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
            "score": [1, 1, 1],
        }
    )

    return BEIRDataset(corpus=corpus, queries=queries, qrels=qrels)


@pytest.fixture
def evaluator(mock_vision_retriever):
    return ViDoReEvaluatorBEIR(vision_retriever=mock_vision_retriever)


def test_init(evaluator):
    assert isinstance(evaluator.vision_retriever, BaseVisionRetriever)


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
