from unittest.mock import Mock

import pytest
import torch
from datasets import Dataset
from PIL import Image

from vidore_benchmark.compression.token_pooling import BaseEmbeddingPooler
from vidore_benchmark.evaluation.evaluate import evaluate_dataset
from vidore_benchmark.retrievers.bm25_retriever import BM25Retriever
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever

EMBEDDING_DIM = 32


@pytest.fixture
def mock_vision_retriever():
    retriever = Mock(spec=VisionRetriever)
    retriever.use_visual_embedding = True

    # Mock the scoring methods
    retriever.forward_queries.return_value = torch.rand(2, EMBEDDING_DIM)
    retriever.forward_passages.return_value = torch.rand(3, EMBEDDING_DIM)
    retriever.get_scores.return_value = torch.tensor(
        [
            [0.8, 0.6, 0.3],
            [0.7, 0.5, 0.4],
        ]
    )

    # Mock the results processing methods
    retriever.get_relevant_docs_results.return_value = (
        {"query1": [0], "query2": [1]},  # relevant_docs
        {"query1": [0, 1, 2], "query2": [0, 1, 2]},  # results
    )

    retriever.compute_metrics.return_value = {"ndcg": 0.85, "map": 0.75, "recall": 0.90}

    return retriever


@pytest.fixture
def mock_dataset():
    # Create a small mock dataset with required columns
    dummy_image = Image.new("RGB", (16, 16), color="black")
    return Dataset.from_dict(
        {
            "query": ["what is in the image?", "describe the scene"],
            "image": [dummy_image] * 2,
            "image_filename": ["img1.jpg", "img2.jpg"],
            "text_description": ["A cat on a mat", "A dog in the park"],
        }
    )


@pytest.fixture
def mock_pooler():
    pooler = Mock(spec=BaseEmbeddingPooler)
    pooler.pool_embeddings.return_value = (torch.rand(16), None)  # Return pooled embedding and None mask
    return pooler


def test_evaluate_dataset_basic(mock_vision_retriever, mock_dataset):
    metrics = evaluate_dataset(
        vision_retriever=mock_vision_retriever,
        ds=mock_dataset,
        batch_query=2,
        batch_passage=2,
    )

    # Verify the expected calls
    mock_vision_retriever.forward_queries.assert_called_once()
    mock_vision_retriever.forward_passages.assert_called()
    mock_vision_retriever.get_scores.assert_called_once()
    mock_vision_retriever.get_relevant_docs_results.assert_called_once()
    mock_vision_retriever.compute_metrics.assert_called_once()

    # Check if metrics are returned correctly
    assert isinstance(metrics, dict)
    assert "ndcg" in metrics
    assert "map" in metrics
    assert "recall" in metrics


def test_evaluate_dataset_with_pooler(mock_vision_retriever, mock_dataset, mock_pooler):
    metrics = evaluate_dataset(
        vision_retriever=mock_vision_retriever,
        ds=mock_dataset,
        batch_query=2,
        batch_passage=2,
        embedding_pooler=mock_pooler,
    )

    # Verify pooler was called for each passage
    mock_pooler.pool_embeddings.assert_called()
    assert isinstance(metrics, dict)


def test_evaluate_dataset_with_bm25(mock_dataset):
    bm25_retriever = Mock(spec=BM25Retriever)
    bm25_retriever.use_visual_embedding = False

    # Mock the scoring methods
    bm25_retriever.get_scores_bm25.return_value = torch.tensor(
        [
            [0.8, 0.6, 0.3],
            [0.7, 0.5, 0.4],
        ]
    )
    bm25_retriever.get_relevant_docs_results.return_value = (
        {"query1": [0], "query2": [1]},
        {"query1": [0, 1, 2], "query2": [0, 1, 2]},
    )
    bm25_retriever.compute_metrics.return_value = {"ndcg": 0.80, "map": 0.70, "recall": 0.85}

    metrics = evaluate_dataset(
        vision_retriever=bm25_retriever,
        ds=mock_dataset,
        batch_query=2,
        batch_passage=2,
    )

    assert isinstance(metrics, dict)
    bm25_retriever.get_scores_bm25.assert_called_once()
