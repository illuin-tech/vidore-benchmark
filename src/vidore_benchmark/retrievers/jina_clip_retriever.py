from __future__ import annotations

from typing import List, cast

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel

from vidore_benchmark.retrievers.utils.register_retriever import register_vision_retriever
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever
from vidore_benchmark.utils.iter_utils import batched
from vidore_benchmark.utils.torch_utils import get_torch_device


@register_vision_retriever("jinaai/jina-clip-v1")
class JinaClipRetriever(VisionRetriever):
    def __init__(self, device: str = "auto"):
        super().__init__()
        self.device = get_torch_device(device)
        self.model = AutoModel.from_pretrained("jinaai/jina-clip-v1", trust_remote_code=True).to(self.device)
        self.emb_dim_query = 768
        self.emb_dim_doc = 768

    @property
    def use_visual_embedding(self) -> bool:
        return True

    def forward_queries(self, queries: List[str], **kwargs) -> torch.Tensor:
        output = self.model.encode_text(queries)
        return torch.tensor(output).to(self.device)

    def forward_documents(self, documents: List[Image.Image], **kwargs) -> torch.Tensor:
        output = self.model.encode_image(documents)
        return torch.tensor(output).to(self.device)

    def get_scores(
        self,
        queries: List[str],
        documents: List[Image.Image] | List[str],
        batch_query: int,
        batch_doc: int,
        **kwargs,
    ) -> torch.Tensor:

        # Sanity check: `documents` must be a list of images
        if documents and not all(isinstance(doc, Image.Image) for doc in documents):
            raise ValueError("Documents must be a list of images")
        documents = cast(List[Image.Image], documents)

        list_emb_queries: List[torch.Tensor] = []
        for query_batch in tqdm(batched(queries, batch_query), desc="Query batch", total=len(queries) // batch_query):
            query_batch = cast(List[str], query_batch)
            query_embeddings = self.forward_queries(query_batch)
            list_emb_queries.append(query_embeddings)

        list_emb_documents: List[torch.Tensor] = []
        for doc_batch in tqdm(batched(documents, batch_doc), desc="Document batch", total=len(documents) // batch_doc):
            doc_batch = cast(List[Image.Image], doc_batch)
            doc_embeddings = self.forward_documents(doc_batch)
            list_emb_documents.append(doc_embeddings)

        emb_queries = torch.cat(list_emb_queries, dim=0)
        emb_documents = torch.cat(list_emb_documents, dim=0)

        scores = torch.einsum("bd,cd->bc", emb_queries, emb_documents)

        return scores
