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


def test_deduplicate_dataset_rows(evaluator):
    # Test normal deduplication
    ds = Dataset.from_dict(
        {
            "query": ["q1", "q2", "q1", None, "q3", "q2", None],
            "other_col": [1, 2, 3, 4, 5, 6, 7],
        }
    )

    deduped_ds = evaluator._deduplicate_dataset_rows(ds, "query")

    assert len(deduped_ds) == 3  # Should only have q1, q2, q3
    assert list(deduped_ds["query"]) == ["q1", "q2", "q3"]
    assert list(deduped_ds["other_col"]) == [1, 2, 5]

    # Test dataset with all None values
    ds_all_none = Dataset.from_dict(
        {
            "query": [None, None, None],
            "other_col": [1, 2, 3],
        }
    )
    deduped_all_none = evaluator._deduplicate_dataset_rows(ds_all_none, "query")
    assert len(deduped_all_none) == 0


def test_get_retrieval_results(evaluator, mock_dataset):
    deduped_ds = evaluator._deduplicate_dataset_rows(mock_dataset, "query")
    scores = torch.tensor([[0.5, 0.3, 0.2, 0.1], [0.4, 0.6, 0.3, 0.2]])

    results = evaluator._get_retrieval_results(
        ds_passages=mock_dataset,
        ds_deduped_queries=deduped_ds,
        scores=scores,
    )

    assert len(results) == 2
    assert all(isinstance(query_results, dict) for query_results in results.values())
    assert all(len(query_results) == 4 for query_results in results.values())
    assert all(isinstance(score, float) for query_results in results.values() for score in query_results.values())
    assert all(filename.endswith(".jpg") for query_results in results.values() for filename in query_results.keys())


def test_get_qrels_from_qa_dataset(evaluator, mock_dataset):
    ds_queries = mock_dataset.remove_columns(
        [col for col in mock_dataset.column_names if col != evaluator.query_column]
    )
    ds_deduped_queries = evaluator._deduplicate_dataset_rows(ds=ds_queries, target_column=evaluator.query_column)

    qrels = evaluator._get_qrels_from_qa_dataset(ds=mock_dataset, ds_deduped_queries=ds_deduped_queries)

    # Check structure and content
    assert isinstance(qrels, dict)
    assert len(qrels) == 2  # Only non-None queries should be included
    assert all(isinstance(query_qrels, dict) for query_qrels in qrels.values())
    assert all(score == 1 for query_qrels in qrels.values() for score in query_qrels.values())
    assert all(filename.endswith(".jpg") for query_qrels in qrels.values() for filename in query_qrels.keys())

    # Check specific mappings
    assert "query1" in qrels
    assert "img1.jpg" in qrels["query1"]
    assert "query2" in qrels
    assert "img2.jpg" in qrels["query2"]


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
