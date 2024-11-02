from typing import Generator

import pytest

from vidore_benchmark.retrievers.bm25_retriever import BM25Retriever


@pytest.fixture(scope="module")
def retriever() -> Generator[BM25Retriever, None, None]:
    yield BM25Retriever()


@pytest.mark.slow
def test_forward_queries(retriever: BM25Retriever, queries_fixture):
    with pytest.raises(NotImplementedError):
        _ = retriever.forward_queries(queries_fixture, batch_size=1)


@pytest.mark.slow
def test_forward_documents(retriever: BM25Retriever, text_passage_fixture):
    with pytest.raises(NotImplementedError):
        _ = retriever.forward_queries(text_passage_fixture, batch_size=1)


@pytest.mark.slow
def test_get_scores(retriever: BM25Retriever, queries_fixture, text_passage_fixture):
    scores = retriever.get_scores_bm25(queries_fixture, text_passage_fixture, batch_query=1, batch_doc=1)
    assert scores.shape == (len(queries_fixture), len(text_passage_fixture))
