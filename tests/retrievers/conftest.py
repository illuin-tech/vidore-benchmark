from typing import List

import pytest
import torch
from PIL import Image

DUMMY_TEXT = "Lorem ipsum"
EMBEDDING_DIM = 16


@pytest.fixture
def queries_fixture() -> List[str]:
    queries = [
        "What is the organizational structure for our R&D department?",
        "Can you provide a breakdown of last year’s financial performance?",
    ]
    return queries


@pytest.fixture
def image_passage_fixture(queries_fixture) -> List[Image.Image]:
    images = [Image.new("RGB", (16, 16), color="black") for _ in queries_fixture]
    return images


@pytest.fixture
def text_passage_fixture(queries_fixture) -> List[str]:
    return [DUMMY_TEXT for _ in queries_fixture]


@pytest.fixture
def query_single_vector_embeddings_fixture() -> List[torch.Tensor]:
    return [
        torch.rand(EMBEDDING_DIM),
        torch.rand(EMBEDDING_DIM),
    ]


@pytest.fixture
def passage_single_vector_embeddings_fixture() -> List[torch.Tensor]:
    return [
        torch.rand(EMBEDDING_DIM),
        torch.rand(EMBEDDING_DIM),
        torch.rand(EMBEDDING_DIM),
    ]


@pytest.fixture
def query_multi_vector_embeddings_fixture() -> List[torch.Tensor]:
    return [
        torch.rand(2, EMBEDDING_DIM),
        torch.rand(4, EMBEDDING_DIM),
    ]


@pytest.fixture
def passage_multi_vector_embeddings_fixture() -> List[torch.Tensor]:
    return [
        torch.rand(8, EMBEDDING_DIM),
        torch.rand(6, EMBEDDING_DIM),
        torch.rand(12, EMBEDDING_DIM),
    ]
