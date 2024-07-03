import pytest
from vidore_benchmark.retrievers.siglip_retriever import SigLIPRetriever


@pytest.fixture
def retriever():
    return SigLIPRetriever()


def test_forward_queries(retriever: SigLIPRetriever, queries_fixtures):
    embedding_queries = retriever.forward_queries(queries_fixtures, batch_size=1)
    assert len(embedding_queries) == len(queries_fixtures)


def test_forward_documents(retriever: SigLIPRetriever, document_images_fixture):
    embedding_docs = retriever.forward_documents(document_images_fixture, batch_size=1)
    assert len(embedding_docs) == len(document_images_fixture)


def test_get_scores(retriever: SigLIPRetriever, queries_fixtures, document_images_fixture):
    emb_query = retriever.forward_queries(queries_fixtures, batch_size=1)
    emb_doc = retriever.forward_documents(document_images_fixture, batch_size=1)
    scores = retriever.get_scores(emb_query, emb_doc)
    assert scores.shape == (len(queries_fixtures), len(document_images_fixture))
