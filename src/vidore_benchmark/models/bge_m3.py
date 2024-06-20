from typing import Any, Dict, List

import torch
from FlagEmbedding import BGEM3FlagModel

from vidore_benchmark.dataset.vision_collator import VisionCollator
from vidore_benchmark.models.utils.register_models import register_collator, register_text_retriever
from vidore_benchmark.models.vision_retriever import VisionRetriever


@register_text_retriever("BAAI/bge-m3")
class BGEM3(VisionRetriever):
    def __init__(self, model: BGEM3FlagModel, processor: None, collator: VisionCollator, *args, **kwargs):
        self.model = model
        self.processor = processor
        self.collator = collator
        self.is_vision_retriever = False
        self.is_multi_vector = False

    def to(self, device: str | torch.device) -> VisionRetriever:
        raise NotImplementedError

    def forward_queries(self, queries, **kwargs) -> torch.Tensor:
        output = self.model.encode(queries, max_length=512)["dense_vecs"]
        return torch.tensor(output)

    def forward_documents(self, documents: List[str], **kwargs) -> torch.Tensor:
        output = self.model.encode(documents)["dense_vecs"]
        return torch.tensor(output)

    def embed_queries(self, queries: List[str]) -> torch.Tensor:
        return torch.tensor(queries)

    def embed_documents(self, documents) -> torch.Tensor:
        return torch.tensor(documents)


@register_collator("BAAI/bge-m3")
class CollatorBGEM3(VisionCollator):
    def __init__(self):
        self.col_document: str = "text_description"
        self.col_query: str = "query"

    def __call__(self, batch: Dict[str, List[Dict[str, torch.Tensor]]]) -> Any:
        # queries = [item[self.col_query] for item in batch]
        documents = [item[self.col_document] for item in batch]
        return {"document": documents}
