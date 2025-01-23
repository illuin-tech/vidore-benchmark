from typing import Generator

import pytest

from vidore_benchmark.retrievers.colidefics3_retriever import ColIdefics3Retriever
from vidore_benchmark.utils.torch_utils import tear_down_torch


@pytest.fixture(scope="module")
def retriever() -> Generator[ColIdefics3Retriever, None, None]:
    yield ColIdefics3Retriever(pretrained_model_name_or_path="vidore/colSmol-256M")
    tear_down_torch()


@pytest.mark.slow
def test_forward_queries(retriever: ColIdefics3Retriever, queries_fixture):
    embedding_queries = retriever.forward_queries(queries_fixture, batch_size=1)
    assert len(embedding_queries) == len(queries_fixture)


@pytest.mark.slow
def test_forward_documents(retriever: ColIdefics3Retriever, image_passage_fixture):
    embedding_docs = retriever.forward_passages(image_passage_fixture, batch_size=1)
    assert len(embedding_docs) == len(image_passage_fixture)


@pytest.mark.slow
def test_get_scores(
    retriever: ColIdefics3Retriever,
    query_multi_vector_embeddings_fixture,
    passage_multi_vector_embeddings_fixture,
):
    scores = retriever.get_scores(query_multi_vector_embeddings_fixture, passage_multi_vector_embeddings_fixture)
    assert scores.shape == (len(query_multi_vector_embeddings_fixture), len(passage_multi_vector_embeddings_fixture))
