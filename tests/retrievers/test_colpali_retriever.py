import pytest
from vidore_benchmark.retrievers.colpali_retriever import ColPaliRetriever


@pytest.fixture
def retriever():
    return ColPaliRetriever()


def test_forward_queries(retriever: ColPaliRetriever, queries_fixtures):
    embedding_queries = retriever.forward_queries(queries_fixtures)
    assert len(embedding_queries) == len(queries_fixtures)
    assert embedding_queries[0].shape[1] == 128


def test_forward_documents(retriever: ColPaliRetriever, document_images_fixture):
    embedding_docs = retriever.forward_documents(document_images_fixture)
    assert len(embedding_docs) == len(document_images_fixture)
    assert embedding_docs[0].shape[1] == 128


def test_get_scores(retriever: ColPaliRetriever, queries_fixtures, document_images_fixture):
    scores = retriever.get_scores(queries_fixtures, document_images_fixture, batch_query=1, batch_doc=1)
    assert scores.shape == (len(queries_fixtures), len(document_images_fixture))
