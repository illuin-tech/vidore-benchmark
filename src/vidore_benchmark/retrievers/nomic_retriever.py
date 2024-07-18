from __future__ import annotations

import math
from typing import List, Optional, cast

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

from vidore_benchmark.retrievers.utils.register_retriever import register_vision_retriever
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever
from vidore_benchmark.utils.iter_utils import batched
from vidore_benchmark.utils.torch_utils import get_torch_device


@register_vision_retriever("nomic-ai/nomic-embed-vision-v1.5")
class NomicVisionRetriever(VisionRetriever):
    def __init__(self, device: str = "auto"):
        super().__init__()
        self.device = get_torch_device(device)

        self.model = AutoModel.from_pretrained("nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True).to(
            self.device
        )
        self.processor = AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5")

        self.text_model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True).to(
            self.device
        )
        self.text_tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
        self.emb_dim_query = 768
        self.emb_dim_doc = 768

    @property
    def use_visual_embedding(self) -> bool:
        return True

    @staticmethod
    def _mean_pooling(model_output: Tensor, attention_mask: Tensor) -> Tensor:
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward_queries(self, queries, batch_size: int, **kwargs) -> List[torch.Tensor]:
        list_emb_queries: List[torch.Tensor] = []
        for query_batch in tqdm(
            batched(queries, batch_size), desc="Query batch", total=math.ceil(len(queries) / batch_size)
        ):
            query_batch = cast(List[str], query_batch)

            query_texts = ["search_query: " + query for query in query_batch]
            encoded_input = self.text_tokenizer(query_texts, padding=True, truncation=True, return_tensors="pt").to(
                self.device
            )
            with torch.no_grad():
                qs = self.text_model(**encoded_input)
            qs = self._mean_pooling(qs, encoded_input["attention_mask"])  # type: ignore
            qs = F.layer_norm(qs, normalized_shape=(qs.shape[1],))
            qs = F.normalize(qs, p=2, dim=1)

            query_embeddings = torch.tensor(qs).to(self.device)
            list_emb_queries.append(query_embeddings)

        return list_emb_queries

    def forward_documents(self, documents, batch_size: int, **kwargs) -> List[torch.Tensor]:
        list_emb_documents: List[torch.Tensor] = []
        for doc_batch in tqdm(
            batched(documents, batch_size), desc="Document batch", total=math.ceil(len(documents) / batch_size)
        ):
            doc_batch = cast(List[Image.Image], doc_batch)

            vision_inputs = self.processor(doc_batch, return_tensors="pt").to(self.device)
            with torch.no_grad():
                ps = self.model(**vision_inputs).last_hidden_state
                ps = F.normalize(ps[:, 0], p=2, dim=1)

            doc_embeddings = torch.tensor(ps).to(self.device)
            list_emb_documents.append(doc_embeddings)

        return list_emb_documents

    def get_scores(
        self,
        list_emb_queries: List[torch.Tensor],
        list_emb_documents: List[torch.Tensor],
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        emb_queries = torch.cat(list_emb_queries, dim=0)
        emb_documents = torch.cat(list_emb_documents, dim=0)

        scores = torch.einsum("bd,cd->bc", emb_queries, emb_documents)

        return scores
