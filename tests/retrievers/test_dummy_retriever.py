import pytest
from vidore_benchmark.retrievers.dummy_retriever import DummyRetriever


@pytest.fixture
def dummy_retriever():
    return DummyRetriever()


def test_forward_queries(dummy_retriever: DummyRetriever, queries_fixtures):
    embeddings_queries = dummy_retriever.forward_queries(queries_fixtures)
    assert embeddings_queries.shape == (len(queries_fixtures), dummy_retriever.emb_dim_query)


def test_forward_documents(dummy_retriever: DummyRetriever, document_filepaths_fixture):
    embeddings_docs = dummy_retriever.forward_documents(document_filepaths_fixture)
    assert embeddings_docs.shape == (len(document_filepaths_fixture), dummy_retriever.emb_dim_doc)


def test_get_scores(dummy_retriever: DummyRetriever, queries_fixtures, document_filepaths_fixture):
    scores = dummy_retriever.get_scores(queries_fixtures, document_filepaths_fixture, batch_query=1, batch_doc=1)
    assert scores.shape == (len(queries_fixtures), len(document_filepaths_fixture))
