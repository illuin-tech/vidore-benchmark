from typing import Generator

import pytest

from vidore_benchmark.retrievers.siglip_retriever import SigLIPRetriever
from vidore_benchmark.utils.testing_utils import tear_down_torch


@pytest.fixture(scope="module")
def retriever() -> Generator[SigLIPRetriever, None, None]:
    yield SigLIPRetriever()
    tear_down_torch()


@pytest.mark.slow
def test_forward_queries(retriever: SigLIPRetriever, queries_fixture):
    embedding_queries = retriever.forward_queries(queries_fixture, batch_size=1)
    assert len(embedding_queries) == len(queries_fixture)


@pytest.mark.slow
def test_forward_documents(retriever: SigLIPRetriever, image_passage_fixture):
    embedding_docs = retriever.forward_passages(image_passage_fixture, batch_size=1)
    assert len(embedding_docs) == len(image_passage_fixture)


@pytest.mark.slow
def test_get_scores(retriever: SigLIPRetriever, queries_fixture, image_passage_fixture):
    query_embeddings = retriever.forward_queries(queries_fixture, batch_size=1)
    passage_embeddings = retriever.forward_passages(image_passage_fixture, batch_size=1)
    scores = retriever.get_scores(query_embeddings, passage_embeddings)
    assert scores.shape == (len(queries_fixture), len(image_passage_fixture))
