import pytest
from vidore_benchmark.retrievers.bm25_retriever import BM25Retriever


@pytest.fixture
def retriever():
    return BM25Retriever()


def test_forward_queries(retriever: BM25Retriever, queries_fixtures):
    with pytest.raises(NotImplementedError):
        embeddings_queries = retriever.forward_queries(queries_fixtures)


def test_forward_documents(retriever: BM25Retriever, document_ocr_text_fixture):
    with pytest.raises(NotImplementedError):
        embeddings_docs = retriever.forward_queries(document_ocr_text_fixture)


def test_get_scores(retriever: BM25Retriever, queries_fixtures, document_ocr_text_fixture):
    scores = retriever.get_scores(queries_fixtures, document_ocr_text_fixture, batch_query=1, batch_doc=1)
    assert scores.shape == (len(queries_fixtures), len(document_ocr_text_fixture))
