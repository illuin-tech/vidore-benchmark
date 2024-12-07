from typing import Generator

import pytest

from vidore_benchmark.retrievers.bge_m3_retriever import BGEM3Retriever
from vidore_benchmark.utils.torch_utils import tear_down_torch


@pytest.fixture(scope="module")
def retriever() -> Generator[BGEM3Retriever, None, None]:
    yield BGEM3Retriever()
    tear_down_torch()


@pytest.mark.slow
def test_forward_queries(retriever: BGEM3Retriever, queries_fixture):
    embeddings_queries = retriever.forward_queries(queries_fixture, batch_size=1)
    assert len(embeddings_queries) == len(queries_fixture)


@pytest.mark.slow
def test_forward_documents(retriever: BGEM3Retriever, text_passage_fixture):
    embeddings_docs = retriever.forward_passages(text_passage_fixture, batch_size=1)
    assert len(embeddings_docs) == len(text_passage_fixture)


@pytest.mark.slow
def test_get_scores(
    retriever: BGEM3Retriever,
    query_single_vector_embeddings_fixture,
    passage_single_vector_embeddings_fixture,
):
    scores = retriever.get_scores(query_single_vector_embeddings_fixture, passage_single_vector_embeddings_fixture)
    assert scores.shape == (len(query_single_vector_embeddings_fixture), len(passage_single_vector_embeddings_fixture))
