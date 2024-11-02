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
def test_forward_queries(retriever: JinaClipRetriever, queries_fixture):
    embeddings_queries = retriever.forward_queries(queries_fixture, batch_size=1)
    assert len(embeddings_queries) == len(queries_fixture)


@pytest.mark.slow
def test_forward_documents(retriever: JinaClipRetriever, image_passage_fixture):
    embeddings_docs = retriever.forward_passages(image_passage_fixture, batch_size=1)
    assert len(embeddings_docs) == len(image_passage_fixture)


@pytest.mark.slow
def test_get_scores(retriever: JinaClipRetriever, queries_fixture, image_passage_fixture):
    emb_query = retriever.forward_queries(queries_fixture, batch_size=1)
    emb_doc = retriever.forward_passages(image_passage_fixture, batch_size=1)
    scores = retriever.get_scores(emb_query, emb_doc)
    assert scores.shape == (len(queries_fixture), len(image_passage_fixture))
