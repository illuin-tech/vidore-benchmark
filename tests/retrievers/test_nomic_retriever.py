import pytest
from vidore_benchmark.retrievers.nomic_retriever import NomicVisionRetriever


@pytest.fixture
def retriever():
    return NomicVisionRetriever()


def test_forward_queries(retriever: NomicVisionRetriever, queries_fixtures):
    embeddings_queries = retriever.forward_queries(queries_fixtures, batch_size=1)
    assert len(embeddings_queries) == len(queries_fixtures)


def test_forward_documents(retriever: NomicVisionRetriever, document_images_fixture):
    embeddings_docs = retriever.forward_documents(document_images_fixture, batch_size=1)
    assert len(embeddings_docs) == len(document_images_fixture)


def test_get_scores(retriever: NomicVisionRetriever, queries_fixtures, document_images_fixture):
    emb_query = retriever.forward_queries(queries_fixtures, batch_size=1)
    emb_doc = retriever.forward_documents(document_images_fixture, batch_size=1)
    scores = retriever.get_scores(emb_query, emb_doc)
    assert scores.shape == (len(queries_fixtures), len(document_images_fixture))
