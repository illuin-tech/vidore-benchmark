from __future__ import annotations

import math
from typing import List, Optional, cast

import torch
from PIL import Image
from tqdm import tqdm

from vidore_benchmark.retrievers.utils.register_retriever import register_vision_retriever
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever
from vidore_benchmark.utils.iter_utils import batched

# soft deps to cohere
try:
    import cohere
except ImportError:
    cohere = None


@register_vision_retriever("cohere/embed-v3")
class CohereAPIRetriever(VisionRetriever):
    def __init__(self):
        super().__init__()
        # Initialize the Cohere client with env variable COHERE_API_KEY
        # api_key = os.getenv("COHERE_API_KEY")
        api_key = "izJl1Jkk2fOkg4ZaMcUHlXXA8p13MUKeoNSvN9LG"
        self.co = cohere.ClientV2(api_key)

    @property
    def use_visual_embedding(self) -> bool:
        return True

    def forward_queries(self, queries, batch_size: int, **kwargs) -> List[List[float]]:
        list_emb_queries: List[List[float]] = []
        print(f"queries: {len(queries)}")
        for query_batch in tqdm(
            batched(queries, batch_size), desc="Query batch", total=math.ceil(len(queries) / batch_size)
        ):
            # delay
            # time.sleep(2)
            response = self.co.embed(
                texts=query_batch, model="embed-english-v3.0", input_type="search_query",
                embedding_types=["float"]
            )
            query_embeddings = list(response.embeddings.float_)

            list_emb_queries.extend(query_embeddings)

        print(f"list_emb_queries: {len(list_emb_queries)}")
        return list_emb_queries

    def forward_documents(self, documents, batch_size: int, **kwargs) -> List[List[float]]:
        list_emb_documents: List[List[float]] = []
        # documents = documents[:4]
        for doc_batch in tqdm(
            batched(documents, batch_size), desc="Document batch", total=math.ceil(len(documents) / batch_size)
        ):
            doc_batch = cast(List[Image.Image], doc_batch)

            def _to_b64(image: Image.Image) -> str:
                import base64
                from io import BytesIO

                buffer = BytesIO()
                image.save(buffer, format="JPEG")
                stringified_buffer = base64.b64encode(buffer.getvalue()).decode('utf-8')
                content_type = "image/jpeg"
                image_base64 = f"data:{content_type};base64,{stringified_buffer}"
                return image_base64

            images_base64 = [_to_b64(doc) for doc in doc_batch]

            # delay
            import time
            time.sleep(2)

            response = self.co.embed(
                model="embed-english-v3.0",
                input_type="image",
                embedding_types=["float"],
                images=images_base64
            )
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