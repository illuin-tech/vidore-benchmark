from typing import Generator

import pytest
from vidore_benchmark.retrievers.cohere_api_retriever import CohereAPIRetriever
from vidore_benchmark.utils.testing_utils import tear_down_torch


@pytest.fixture(scope="module")
def retriever() -> Generator[CohereAPIRetriever, None, None]:
    yield CohereAPIRetriever(pretrained_model_name_or_path="embed-english-v3.0")
    tear_down_torch()


@pytest.mark.slow
def test_forward_queries(retriever: CohereAPIRetriever, queries_fixture):
    embedding_queries = retriever.forward_queries(queries_fixture, batch_size=1)
    assert len(embedding_queries) == len(queries_fixture)


@pytest.mark.slow
def test_forward_documents(retriever: CohereAPIRetriever, document_images_fixture):
    embedding_docs = retriever.forward_documents(document_images_fixture, batch_size=1)
    assert len(embedding_docs) == len(document_images_fixture)


@pytest.mark.slow
def test_get_scores(retriever: CohereAPIRetriever, queries_fixture, document_images_fixture):
    emb_query = retriever.forward_queries(queries_fixture, batch_size=1)
    emb_doc = retriever.forward_documents(document_images_fixture, batch_size=1)
    scores = retriever.get_scores(emb_query, emb_doc)
    assert scores.shape == (len(queries_fixture), len(document_images_fixture))
