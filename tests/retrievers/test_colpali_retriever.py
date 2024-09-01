from typing import Generator

import pytest

from vidore_benchmark.retrievers.colpali_retriever import ColPaliRetriever
from vidore_benchmark.utils.testing_utils import tear_down_torch


@pytest.fixture(scope="module")
def retriever() -> Generator[ColPaliRetriever, None, None]:
    yield ColPaliRetriever()
    tear_down_torch()


@pytest.mark.slow
def test_forward_queries(retriever: ColPaliRetriever, queries_fixtures):
    embedding_queries = retriever.forward_queries(queries_fixtures, batch_size=1)
    assert len(embedding_queries) == len(queries_fixtures)
    assert embedding_queries[0].shape[1] == 128


@pytest.mark.slow
def test_forward_documents(retriever: ColPaliRetriever, document_images_fixture):
    embedding_docs = retriever.forward_documents(document_images_fixture, batch_size=1)
    assert len(embedding_docs) == len(document_images_fixture)
    assert embedding_docs[0].shape[1] == 128


@pytest.mark.slow
def test_get_scores(retriever: ColPaliRetriever, queries_fixtures, document_images_fixture):
    emb_query = retriever.forward_queries(queries_fixtures, batch_size=1)
    emb_doc = retriever.forward_documents(document_images_fixture, batch_size=1)
    scores = retriever.get_scores(emb_query, emb_doc)
    assert scores.shape == (len(queries_fixtures), len(document_images_fixture))
