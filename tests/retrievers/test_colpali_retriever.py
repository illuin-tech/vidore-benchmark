from typing import Generator

import pytest

from vidore_benchmark.retrievers.colpali_retriever import ColPaliRetriever
from vidore_benchmark.utils.testing_utils import tear_down_torch


@pytest.fixture(scope="module")
def retriever() -> Generator[ColPaliRetriever, None, None]:
    yield ColPaliRetriever(pretrained_model_name_or_path="vidore/colpali-v1.2-merged")
    tear_down_torch()


@pytest.mark.slow
def test_forward_queries(retriever: ColPaliRetriever, queries_fixture):
    embedding_queries = retriever.forward_queries(queries_fixture, batch_size=1)
    assert len(embedding_queries) == len(queries_fixture)
    assert embedding_queries[0].shape[1] == 128


@pytest.mark.slow
def test_forward_documents(retriever: ColPaliRetriever, image_passage_fixture):
    embedding_docs = retriever.forward_passages(image_passage_fixture, batch_size=1)
    assert len(embedding_docs) == len(image_passage_fixture)
    assert embedding_docs[0].shape[1] == 128


@pytest.mark.slow
def test_get_scores(retriever: ColPaliRetriever, queries_fixture, image_passage_fixture):
    query_embeddings = retriever.forward_queries(queries_fixture, batch_size=1)
    passage_embeddings = retriever.forward_passages(image_passage_fixture, batch_size=1)
    scores = retriever.get_scores(query_embeddings, passage_embeddings)
    assert scores.shape == (len(queries_fixture), len(image_passage_fixture))
