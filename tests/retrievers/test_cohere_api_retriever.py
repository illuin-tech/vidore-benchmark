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
def test_forward_documents(retriever: CohereAPIRetriever, image_passage_fixture):
    embedding_docs = retriever.forward_passages(image_passage_fixture, batch_size=1)
    assert len(embedding_docs) == len(image_passage_fixture)


@pytest.mark.slow
def test_get_scores(retriever: CohereAPIRetriever, queries_fixture, image_passage_fixture):
    emb_query = retriever.forward_queries(queries_fixture, batch_size=1)
    emb_doc = retriever.forward_passages(image_passage_fixture, batch_size=1)
    scores = retriever.get_scores(emb_query, emb_doc)
    assert scores.shape == (len(queries_fixture), len(image_passage_fixture))
