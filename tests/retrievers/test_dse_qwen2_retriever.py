from typing import Generator

import pytest

from vidore_benchmark.retrievers.dse_qwen2_retriever import DSEQwen2Retriever
from vidore_benchmark.utils.torch_utils import tear_down_torch


@pytest.fixture(scope="module")
def retriever() -> Generator[DSEQwen2Retriever, None, None]:
    yield DSEQwen2Retriever(pretrained_model_name_or_path="MrLight/dse-qwen2-2b-mrl-v1")
    tear_down_torch()


@pytest.mark.slow
def test_forward_queries(retriever: DSEQwen2Retriever, queries_fixture):
    embedding_queries = retriever.forward_queries(queries_fixture, batch_size=1)
    assert len(embedding_queries) == len(queries_fixture)


@pytest.mark.slow
def test_forward_documents(retriever: DSEQwen2Retriever, image_passage_fixture):
    embedding_docs = retriever.forward_passages(image_passage_fixture, batch_size=1)
    assert len(embedding_docs) == len(image_passage_fixture)


@pytest.mark.slow
def test_get_scores(
    retriever: DSEQwen2Retriever,
    query_single_vector_embeddings_fixture,
    passage_single_vector_embeddings_fixture,
):
    scores = retriever.get_scores(query_single_vector_embeddings_fixture, passage_single_vector_embeddings_fixture)
    assert scores.shape == (len(query_single_vector_embeddings_fixture), len(passage_single_vector_embeddings_fixture))
