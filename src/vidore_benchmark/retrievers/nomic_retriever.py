from __future__ import annotations

from typing import List, cast

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

    def forward_queries(self, queries: List[str], **kwargs) -> torch.Tensor:
        query_texts = ["search_query: " + query for query in queries]
        encoded_input = self.text_tokenizer(query_texts, padding=True, truncation=True, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            qs = self.text_model(**encoded_input)
        qs = self._mean_pooling(qs, encoded_input["attention_mask"])  # type: ignore
        qs = F.layer_norm(qs, normalized_shape=(qs.shape[1],))
        qs = F.normalize(qs, p=2, dim=1)

        return torch.tensor(qs).to(self.device)

    def forward_documents(self, documents: List[Image.Image], **kwargs) -> torch.Tensor:
        vision_inputs = self.processor(documents, return_tensors="pt").to(self.device)
        with torch.no_grad():
            ps = self.model(**vision_inputs).last_hidden_state
            ps = F.normalize(ps[:, 0], p=2, dim=1)

        return torch.tensor(ps).to(self.device)

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
            raise ValueError("Documents must be a list of Pillow images")
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

        assert scores.shape == (emb_queries.shape[0], emb_documents.shape[0])

        return scores
