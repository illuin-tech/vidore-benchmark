from __future__ import annotations

import math
import os
import time
from typing import List, Optional, cast

import torch
from dotenv import load_dotenv
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from vidore_benchmark.retrievers.utils.register_retriever import register_vision_retriever
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever
from vidore_benchmark.utils.iter_utils import batched

try:
    import cohere
except ImportError:
    pass

load_dotenv(override=True)


@register_vision_retriever("cohere")
class CohereAPIRetriever(VisionRetriever):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "embed-english-v3.0",
    ):

        super().__init__()

        api_key = os.getenv("COHERE_API_KEY", None)
        if api_key is None:
            raise ValueError("COHERE_API_KEY environment variable is not set")

        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.co = cohere.ClientV2(api_key)

    @property
    def use_visual_embedding(self) -> bool:
        return True

    @staticmethod
    def convert_image_to_base64(image: Image.Image) -> str:
        import base64
        from io import BytesIO

        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        stringified_buffer = base64.b64encode(buffer.getvalue()).decode("utf-8")
        content_type = "image/jpeg"
        image_base64 = f"data:{content_type};base64,{stringified_buffer}"
        return image_base64

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=10))
    def call_api_queries(self, queries: List[str]):
        response = self.co.embed(
            model=self.pretrained_model_name_or_path,
            input_type="search_query",
            embedding_types=["float"],
            texts=queries,
        )
        return response

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=10))
    def call_api_images(self, images_b64: List[str]):
        response = self.co.embed(
            model="embed-english-v3.0",
            input_type="image",
            embedding_types=["float"],
            images=images_b64,
        )
        return response

    def forward_queries(self, queries: List[str], batch_size: int, **kwargs) -> List[List[float]]:
        list_emb_queries: List[List[float]] = []

        for query_batch in tqdm(
            batched(queries, batch_size), desc="Query batch", total=math.ceil(len(queries) / batch_size)
        ):
            response = self.call_api_queries(query_batch)
            query_embeddings = list(response.embeddings.float_)

            list_emb_queries.extend(query_embeddings)

        return list_emb_queries

    def forward_documents(self, documents, batch_size: int, **kwargs) -> List[List[float]]:
        # NOTE: Batch size should be set to 1 with the current Cohere API.
        list_emb_documents: List[List[float]] = []

        for doc_batch in tqdm(
            batched(documents, batch_size), desc="Document batch", total=math.ceil(len(documents) / batch_size)
        ):
            doc_batch = cast(List[Image.Image], doc_batch)
            images_base64 = [self.convert_image_to_base64(doc) for doc in doc_batch]

            # Optional delay:
            time.sleep(2)

            response = self.call_api_images(images_base64)
            doc_embeddings = list(response.embeddings.float_)

            list_emb_documents.extend(doc_embeddings)

        return list_emb_documents

    def get_scores(
        self,
        list_emb_queries: List[List[float]],
        list_emb_documents: List[List[float]],
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:

        emb_queries = torch.tensor(list_emb_queries)
        emb_documents = torch.tensor(list_emb_documents)
        scores = torch.einsum("bd,cd->bc", emb_queries, emb_documents)
        return scores
