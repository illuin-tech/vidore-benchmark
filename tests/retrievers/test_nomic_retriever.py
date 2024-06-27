import pytest
from vidore_benchmark.retrievers.nomic_retriever import NomicVisionRetriever


@pytest.fixture
def retriever():
    return NomicVisionRetriever()


def test_forward_queries(retriever: NomicVisionRetriever, queries_fixtures):
    embeddings_queries = retriever.forward_queries(queries_fixtures)
    assert embeddings_queries.shape == (len(queries_fixtures), retriever.emb_dim_query)


def test_forward_documents(retriever: NomicVisionRetriever, document_images_fixture):
    embeddings_docs = retriever.forward_documents(document_images_fixture)
    assert embeddings_docs.shape == (len(document_images_fixture), retriever.emb_dim_doc)


def test_get_scores(retriever: NomicVisionRetriever, queries_fixtures, document_images_fixture):
    scores = retriever.get_scores(queries_fixtures, document_images_fixture, batch_query=1, batch_doc=1)
    assert scores.shape == (len(queries_fixtures), len(document_images_fixture))
