import pytest
from vidore_benchmark.retrievers.bge_m3_retriever import BGEM3Retriever


@pytest.fixture
def retriever():
    return BGEM3Retriever()


def test_forward_queries(retriever: BGEM3Retriever, queries_fixtures):
    embeddings_queries = retriever.forward_queries(queries_fixtures)
    assert embeddings_queries.shape == (len(queries_fixtures), retriever.emb_dim_query)


def test_forward_documents(retriever: BGEM3Retriever, document_ocr_text_fixture):
    embeddings_docs = retriever.forward_documents(document_ocr_text_fixture)
    assert embeddings_docs.shape == (len(document_ocr_text_fixture), retriever.emb_dim_doc)


def test_get_scores(retriever: BGEM3Retriever, queries_fixtures, document_ocr_text_fixture):
    scores = retriever.get_scores(queries_fixtures, document_ocr_text_fixture, batch_query=1, batch_doc=1)
    assert scores.shape == (len(queries_fixtures), len(document_ocr_text_fixture))
