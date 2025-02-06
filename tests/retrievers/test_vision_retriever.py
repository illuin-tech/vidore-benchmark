from typing import Generator, cast

import pytest
import torch
from colpali_engine.compression import HierarchicalTokenPooler
from colpali_engine.models import ColIdefics3, ColIdefics3Processor

from vidore_benchmark.retrievers import VisionRetriever
from vidore_benchmark.retrievers.colidefics3_retriever import ColIdefics3Retriever
from vidore_benchmark.utils.torch_utils import get_torch_device, tear_down_torch


@pytest.fixture(scope="module")
def model_name() -> str:
    return "vidore/colSmol-256M"


@pytest.fixture(scope="module")
def model(model_name: str) -> Generator[ColIdefics3, None, None]:
    model = cast(
        ColIdefics3,
        ColIdefics3.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=get_torch_device("auto"),
        ).eval(),
    )
    yield model
    tear_down_torch()


@pytest.fixture(scope="module")
def processor(model_name: str) -> Generator[ColIdefics3Processor, None, None]:
    model_name = "vidore/colSmol-256M"
    processor = cast(ColIdefics3Processor, ColIdefics3Processor.from_pretrained(model_name))
    yield processor


class TestVisionRetrieverIntegration:
    @pytest.fixture(scope="class")
    def retriever(self, model: ColIdefics3, processor: ColIdefics3Processor) -> Generator[VisionRetriever, None, None]:
        vision_retriever = VisionRetriever(
            model=model,
            processor=processor,
        )
        yield vision_retriever

    @pytest.mark.slow
    def test_forward_queries(self, retriever: ColIdefics3Retriever, queries_fixture):
        embedding_queries = retriever.forward_queries(queries_fixture, batch_size=1)
        assert len(embedding_queries) == len(queries_fixture)

    @pytest.mark.slow
    def test_forward_documents(self, retriever: ColIdefics3Retriever, image_passage_fixture):
        embedding_docs = retriever.forward_passages(image_passage_fixture, batch_size=1)
        assert len(embedding_docs) == len(image_passage_fixture)

    @pytest.mark.slow
    def test_get_scores(
        self,
        retriever: ColIdefics3Retriever,
        query_multi_vector_embeddings_fixture,
        passage_multi_vector_embeddings_fixture,
    ):
        scores = retriever.get_scores(query_multi_vector_embeddings_fixture, passage_multi_vector_embeddings_fixture)
        assert scores.shape == (
            len(query_multi_vector_embeddings_fixture),
            len(passage_multi_vector_embeddings_fixture),
        )


class TestVisionRetrieverWithTokenPoolingIntegration:
    @pytest.mark.slow
    def test_token_pooling_shortens_passage_embeddings(
        self,
        model: ColIdefics3,
        processor: ColIdefics3Processor,
        image_passage_fixture,
    ):
        vision_retriever_without_pooling = VisionRetriever(
            model=model,
            processor=processor,
        )
        embeddings = vision_retriever_without_pooling.forward_passages(image_passage_fixture, batch_size=1)

        token_pooler = HierarchicalTokenPooler(pool_factor=3)
        vision_retriever_with_pooling = VisionRetriever(
            model=model,
            processor=processor,
            token_pooler=token_pooler,
        )
        embeddings_pooled = vision_retriever_with_pooling.forward_passages(image_passage_fixture, batch_size=1)

        for embedding, embedding_pooled in zip(embeddings, embeddings_pooled):
            assert embedding_pooled.shape[0] < embedding.shape[0]
