from typing import Generator

import pytest

from vidore_benchmark.retrievers.colqwen_retriever import ColQwenRetriever
from vidore_benchmark.utils.testing_utils import tear_down_torch


@pytest.fixture(scope="module")
def retriever() -> Generator[ColQwenRetriever, None, None]:
    yield ColQwenRetriever(pretrained_model_name_or_path="vidore/colqwen2-v0.1")
    tear_down_torch()


@pytest.mark.slow
def test_forward_queries(retriever: ColQwenRetriever, queries_fixture):
    embedding_queries = retriever.forward_queries(queries_fixture, batch_size=1)
    assert len(embedding_queries) == len(queries_fixture)


@pytest.mark.slow
def test_forward_documents(retriever: ColQwenRetriever, document_images_fixture):
    embedding_docs = retriever.forward_passages(document_images_fixture, batch_size=1)
    assert len(embedding_docs) == len(document_images_fixture)


@pytest.mark.slow
def test_get_scores(retriever: ColQwenRetriever, queries_fixture, document_images_fixture):
    emb_query = retriever.forward_queries(queries_fixture, batch_size=1)
    emb_doc = retriever.forward_passages(document_images_fixture, batch_size=1)
    scores = retriever.get_scores(emb_query, emb_doc)
    assert scores.shape == (len(queries_fixture), len(document_images_fixture))
