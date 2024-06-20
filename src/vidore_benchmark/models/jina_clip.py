from vidore_benchmark.models.vision_retriever import VisionRetriever
import torch
from PIL import Image
from typing import List, Dict, Any
from vidore_benchmark.utils.registration import register_collator, register_vision_retriever
from vidore_benchmark.dataset.vision_collator import VisionCollator

@register_vision_retriever("jinaai/jina-clip-v1")
class JinaClip(VisionRetriever):
    def __init__(self, model: torch.nn.Module, processor: None, collator: VisionCollator, *args, **kwargs):
        self.model = model
        self.processor = processor
        self.collator = collator
        self.is_vision_retriever = True
        self.is_multi_vector = False

    def to(self, device: str | torch.device) -> VisionRetriever:
        self.model.to(device)
        return self

    def forward_queries(self, queries: List[str], **kwargs) -> torch.Tensor:
        output = self.model.encode_text(queries)
        return torch.tensor(output)

    def forward_documents(self, documents: List[Image.Image], **kwargs) -> torch.Tensor:
        output = self.model.encode_image(documents)
        return torch.tensor(output)

    def embed_queries(self, queries: List[str]) -> torch.Tensor:
        raise NotImplementedError
    def embed_documents(self, documents) -> torch.Tensor:
        raise NotImplementedError

@register_collator("jinaai/jina-clip-v1")
class CollatorJinaClip(VisionCollator):
    def __init__(self):
        self.col_document: str = "image"
        self.col_query: str = "query"

    def __call__(self, batch: Dict[str, List[Dict[str, torch.Tensor]]]) -> Any:
        documents = [item[self.col_document] for item in batch]
        return {"document": documents}