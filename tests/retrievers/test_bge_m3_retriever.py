import pytest
from vidore_benchmark.retrievers.bge_m3_retriever import BGEM3Retriever


@pytest.fixture
def retriever():
    return BGEM3Retriever()


def test_forward_queries(retriever: BGEM3Retriever, queries_fixtures):
    embeddings_queries = retriever.forward_queries(queries_fixtures, batch_size=1)
    assert len(embeddings_queries) == len(queries_fixtures)


def test_forward_documents(retriever: BGEM3Retriever, document_ocr_text_fixture):
    embeddings_docs = retriever.forward_documents(document_ocr_text_fixture, batch_size=1)
    assert len(embeddings_docs) == len(document_ocr_text_fixture)


def test_get_scores(retriever: BGEM3Retriever, queries_fixtures, document_ocr_text_fixture):
    emb_query = retriever.forward_queries(queries_fixtures, batch_size=1)
    emb_doc = retriever.forward_documents(document_ocr_text_fixture, batch_size=1)
    scores = retriever.get_scores(emb_query, emb_doc)
    assert scores.shape == (len(queries_fixtures), len(document_ocr_text_fixture))
