from typing import Generator

import pytest

from vidore_benchmark.retrievers.dummy_vision_retriever import DummyVisionRetriever
from vidore_benchmark.utils.torch_utils import tear_down_torch


@pytest.fixture(scope="module")
def retriever() -> Generator[DummyVisionRetriever, None, None]:
    yield DummyVisionRetriever()
    tear_down_torch()


def test_forward_queries(retriever: DummyVisionRetriever, queries_fixture):
    embeddings_queries = retriever.forward_queries(queries_fixture, batch_size=1)
    assert len(embeddings_queries) == len(queries_fixture)


def test_forward_documents(retriever: DummyVisionRetriever, image_passage_fixture):
    embeddings_docs = retriever.forward_passages(image_passage_fixture, batch_size=1)
    assert len(embeddings_docs) == len(image_passage_fixture)


def test_get_scores(
    retriever: DummyVisionRetriever,
    query_single_vector_embeddings_fixture,
    passage_single_vector_embeddings_fixture,
):
    scores = retriever.get_scores(query_single_vector_embeddings_fixture, passage_single_vector_embeddings_fixture)
    assert scores.shape == (len(query_single_vector_embeddings_fixture), len(passage_single_vector_embeddings_fixture))
