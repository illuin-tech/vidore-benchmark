from vidore_benchmark.models.vision_retriever import VisionRetriever
from FlagEmbedding import BGEM3FlagModel
import torch
from PIL import Image
from typing import List, Dict, Any
from vidore_benchmark.utils.registration import register_collator, register_text_retriever
from vidore_benchmark.dataset.vision_collator import VisionCollator


@register_text_retriever("BGEM3")
class BGEM3(VisionRetriever):
    def __init__(self, model: BGEM3FlagModel, processor: None, collator: VisionCollator, *args, **kwargs):
        self.model = model
        self.processor = processor
        self.collator = collator

    def to(self, device: str | torch.device) -> VisionRetriever:
        raise NotImplementedError

    def forward_queries(self, queries, **kwargs) -> torch.Tensor:
        # No forward pass needed
        output = self.model.encode(queries)
        return torch.tensor(output)

    def forward_documents(self, documents : List[str], **kwargs) -> torch.Tensor:
        # No forward pass needed
        output = self.model.encode(documents)
        return torch.tensor(output)

    def embed_queries(self, queries: List[str]) -> torch.Tensor:
        return torch.tensor(queries)

    def embed_documents(self, documents) -> torch.Tensor:
        return torch.tensor(documents)


@register_collator("BGEM3")
class CollatorBGEM3(VisionCollator):
    def __init__(self):
        self.col_document :str = 'text_description'
        self.col_query: str = 'query'
        print("CollatorBGEM3 initialized")
        pass

    def __call__(self, batch: Dict[str, List[Dict[str, torch.Tensor]]]) -> Any:
        queries = [item[self.col_query] for item in batch]
        documents = [item[self.col_document] for item in batch]
        return {"query": queries, "document": documents}
    