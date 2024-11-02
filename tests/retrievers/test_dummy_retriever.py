from typing import Generator

import pytest

from vidore_benchmark.retrievers.dummy_retriever import DummyRetriever
from vidore_benchmark.utils.testing_utils import tear_down_torch


@pytest.fixture(scope="module")
def retriever() -> Generator[DummyRetriever, None, None]:
    yield DummyRetriever()
    tear_down_torch()


@pytest.mark.slow
def test_forward_queries(retriever: DummyRetriever, queries_fixture):
    embeddings_queries = retriever.forward_queries(queries_fixture, batch_size=1)
    assert len(embeddings_queries) == len(queries_fixture)


@pytest.mark.slow
def test_forward_documents(retriever: DummyRetriever, document_filepaths_fixture):
    embeddings_docs = retriever.forward_passages(document_filepaths_fixture, batch_size=1)
    assert len(embeddings_docs) == len(document_filepaths_fixture)


@pytest.mark.slow
def test_get_scores(retriever: DummyRetriever, queries_fixture, document_filepaths_fixture):
    emb_query = retriever.forward_queries(queries_fixture, batch_size=1)
    emb_doc = retriever.forward_passages(document_filepaths_fixture, batch_size=1)
    scores = retriever.get_scores(emb_query, emb_doc)
    assert scores.shape == (len(queries_fixture), len(document_filepaths_fixture))
