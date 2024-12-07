from typing import Generator

import pytest

from vidore_benchmark.retrievers.colqwen_retriever import ColQwenRetriever
from vidore_benchmark.utils.torch_utils import tear_down_torch


@pytest.fixture(scope="module")
def retriever() -> Generator[ColQwenRetriever, None, None]:
    yield ColQwenRetriever(pretrained_model_name_or_path="vidore/colqwen2-v0.1")
    tear_down_torch()


@pytest.mark.slow
def test_forward_queries(retriever: ColQwenRetriever, queries_fixture):
    embedding_queries = retriever.forward_queries(queries_fixture, batch_size=1)
    assert len(embedding_queries) == len(queries_fixture)


@pytest.mark.slow
def test_forward_documents(retriever: ColQwenRetriever, image_passage_fixture):
    embedding_docs = retriever.forward_passages(image_passage_fixture, batch_size=1)
    assert len(embedding_docs) == len(image_passage_fixture)


@pytest.mark.slow
def test_get_scores(
    retriever: ColQwenRetriever,
    query_multi_vector_embeddings_fixture,
    passage_multi_vector_embeddings_fixture,
):
    scores = retriever.get_scores(query_multi_vector_embeddings_fixture, passage_multi_vector_embeddings_fixture)
    assert scores.shape == (len(query_multi_vector_embeddings_fixture), len(passage_multi_vector_embeddings_fixture))
