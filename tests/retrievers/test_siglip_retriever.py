import pytest
from vidore_benchmark.retrievers.siglip_retriever import SigLIPRetriever


@pytest.fixture
def retriever():
    return SigLIPRetriever()


def test_forward_queries(retriever: SigLIPRetriever, queries_fixtures):
    embedding_queries = retriever.forward_queries(queries_fixtures)
    assert len(embedding_queries) == len(queries_fixtures)
    assert embedding_queries.shape == (len(queries_fixtures), 1152)


def test_forward_documents(retriever: SigLIPRetriever, document_images_fixture):
    embedding_docs = retriever.forward_documents(document_images_fixture)
    assert len(embedding_docs) == len(document_images_fixture)
    assert embedding_docs.shape == (len(document_images_fixture), 1152)


def test_get_scores(retriever: SigLIPRetriever, queries_fixtures, document_images_fixture):
    scores = retriever.get_scores(queries_fixtures, document_images_fixture, batch_query=1, batch_doc=1)
    assert scores.shape == (len(queries_fixtures), len(document_images_fixture))
