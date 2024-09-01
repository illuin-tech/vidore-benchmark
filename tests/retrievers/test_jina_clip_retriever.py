from typing import Generator

import pytest

from vidore_benchmark.retrievers.jina_clip_retriever import JinaClipRetriever
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever
from vidore_benchmark.utils.testing_utils import tear_down_torch


@pytest.fixture(scope="module")
def retriever() -> Generator[VisionRetriever, None, None]:
    yield JinaClipRetriever()
    tear_down_torch()


@pytest.mark.slow
def test_forward_queries(retriever: JinaClipRetriever, queries_fixtures):
    embeddings_queries = retriever.forward_queries(queries_fixtures, batch_size=1)
    assert len(embeddings_queries) == len(queries_fixtures)


@pytest.mark.slow
def test_forward_documents(retriever: JinaClipRetriever, document_images_fixture):
    embeddings_docs = retriever.forward_documents(document_images_fixture, batch_size=1)
    assert len(embeddings_docs) == len(document_images_fixture)


@pytest.mark.slow
def test_get_scores(retriever: JinaClipRetriever, queries_fixtures, document_images_fixture):
    emb_query = retriever.forward_queries(queries_fixtures, batch_size=1)
    emb_doc = retriever.forward_documents(document_images_fixture, batch_size=1)
    scores = retriever.get_scores(emb_query, emb_doc)
    assert scores.shape == (len(queries_fixtures), len(document_images_fixture))
