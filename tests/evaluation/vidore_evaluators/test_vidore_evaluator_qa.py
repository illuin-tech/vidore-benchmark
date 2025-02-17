import pytest
from datasets import Dataset
from PIL import Image

from vidore_benchmark.evaluation.vidore_evaluators.vidore_evaluator_qa import ViDoReEvaluatorQA
from vidore_benchmark.retrievers.base_vision_retriever import BaseVisionRetriever


@pytest.fixture
def mock_qa_dataset():
    return Dataset.from_dict(
        {
            "query": ["query1", "query2", "query1", None],
            "image": [Image.new("RGB", (10, 10)) for _ in range(4)],
            "text_description": ["desc1", "desc2", "desc3", "desc4"],
            "image_filename": ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"],
        }
    )


@pytest.fixture
def evaluator(mock_vision_retriever):
    return ViDoReEvaluatorQA(vision_retriever=mock_vision_retriever)


def test_init(evaluator):
    assert isinstance(evaluator.vision_retriever, BaseVisionRetriever)


def test_evaluate_dataset(evaluator, mock_qa_dataset):
    metrics = evaluator.evaluate_dataset(
        ds=mock_qa_dataset,
        batch_query=2,
        batch_passage=2,
    )

    assert isinstance(metrics, dict)
    assert "ndcg_at_1" in metrics
    assert "map_at_1" in metrics
    assert "recall_at_1" in metrics
    assert "precision_at_1" in metrics
    assert "mrr_at_1" in metrics


def test_evaluate_dataset_with_bm25(mock_bm25_retriever, mock_qa_dataset):
    evaluator = ViDoReEvaluatorQA(vision_retriever=mock_bm25_retriever)
    metrics = evaluator.evaluate_dataset(
        ds=mock_qa_dataset,
        batch_query=2,
        batch_passage=2,
    )

    assert isinstance(metrics, dict)
    assert "ndcg_at_1" in metrics
    assert "map_at_1" in metrics
    assert "recall_at_1" in metrics
    assert "precision_at_1" in metrics
