from __future__ import annotations

import math
import os
import time
from typing import List, Optional, Union, cast

import torch
from dotenv import load_dotenv
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from vidore_benchmark.retrievers.registry_utils import register_vision_retriever
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever
from vidore_benchmark.utils.iter_utils import batched

load_dotenv(override=True)


@register_vision_retriever("cohere")
class CohereAPIRetriever(VisionRetriever):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "embed-english-v3.0",
    ):
        super().__init__()

        try:
            import cohere
        except ImportError:
            raise ImportError(
                'Install the missing dependencies with `pip install "vidore-benchmark[cohere]"` '
                "to use CohereAPIRetriever."
            )

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

    def forward_queries(self, queries: List[str], batch_size: int, **kwargs) -> torch.Tensor:
        list_emb_queries: List[List[float]] = []

        for query_batch in tqdm(
            batched(queries, batch_size),
            desc="Forwarding query batches",
            total=math.ceil(len(queries) / batch_size),
            leave=False,
        ):
            response = self.call_api_queries(query_batch)
            list_emb_queries.extend(list(response.embeddings.float_))

        return torch.tensor(list_emb_queries)

    def forward_passages(self, passages, batch_size: int, **kwargs) -> torch.Tensor:
        # NOTE: Batch size should be set to 1 with the current Cohere API.
        list_emb_passages: List[List[float]] = []

        for passage_batch in tqdm(
            batched(passages, batch_size),
            desc="Forwarding passage batches",
            total=math.ceil(len(passages) / batch_size),
            leave=False,
        ):
            passage_batch = cast(List[Image.Image], passage_batch)
            images_base64 = [self.convert_image_to_base64(doc) for doc in passage_batch]

            # Optional delay:
            time.sleep(2)

            response = self.call_api_images(images_base64)
            list_emb_passages.extend(list(response.embeddings.float_))

        return torch.tensor(list_emb_passages)

    def get_scores(
        self,
        query_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        passage_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Dot-product similarity between queries and passages.
        """
        if isinstance(query_embeddings, list):
            query_embeddings = torch.stack(query_embeddings)
        if isinstance(passage_embeddings, list):
            passage_embeddings = torch.stack(passage_embeddings)

        scores = torch.einsum("bd,cd->bc", query_embeddings, passage_embeddings)
        return scores
