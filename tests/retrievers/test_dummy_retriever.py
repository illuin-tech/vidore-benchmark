import pytest
from vidore_benchmark.retrievers.dummy_retriever import DummyRetriever


@pytest.fixture
def dummy_retriever():
    return DummyRetriever()


def test_forward_queries(dummy_retriever, queries_fixtures):
    embeddings = dummy_retriever.forward_queries(queries_fixtures)
    assert embeddings.shape == (len(queries_fixtures), dummy_retriever.emb_dim_query)


def test_forward_documents(dummy_retriever, documents_fixture):
    embeddings = dummy_retriever.forward_documents(documents_fixture)
    assert embeddings.shape == (len(documents_fixture), dummy_retriever.emb_dim_doc)


def test_get_scores(dummy_retriever, queries_fixtures, documents_fixture):
    scores = dummy_retriever.get_scores(queries_fixtures, documents_fixture, batch_query=1, batch_doc=1)
    assert scores.shape == (len(queries_fixtures), len(documents_fixture))
