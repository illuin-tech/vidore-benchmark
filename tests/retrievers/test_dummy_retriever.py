import pytest
from vidore_benchmark.retrievers.dummy_retriever import DummyRetriever


@pytest.fixture
def dummy_retriever():
    return DummyRetriever()


def test_forward_queries(dummy_retriever: DummyRetriever, queries_fixtures):
    embeddings_queries = dummy_retriever.forward_queries(queries_fixtures, batch_size=1)
    assert len(embeddings_queries) == len(queries_fixtures)

def test_forward_documents(dummy_retriever: DummyRetriever, document_filepaths_fixture):
    embeddings_docs = dummy_retriever.forward_documents(document_filepaths_fixture, batch_size=1)
    assert len(embeddings_docs) == len(document_filepaths_fixture)


def test_get_scores(dummy_retriever: DummyRetriever, queries_fixtures, document_filepaths_fixture):
    emb_query = dummy_retriever.forward_queries(queries_fixtures, batch_size=1)
    emb_doc = dummy_retriever.forward_documents(document_filepaths_fixture, batch_size=1)
    scores = dummy_retriever.get_scores(emb_query, emb_doc)
    assert scores.shape == (len(queries_fixtures), len(document_filepaths_fixture))
