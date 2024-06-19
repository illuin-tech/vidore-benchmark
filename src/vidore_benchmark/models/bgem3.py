from vidore_benchmark.models.vision_retriever import VisionRetriever
from FlagEmbedding import BGEM3FlagModel
import torch
from PIL import Image
from typing import List
from vidore_benchmark.utils.registration import register_vision_retriever, register_collator
from vidore_benchmark.dataset.vision_collator import VisionCollator


@register_vision_retriever("BGEM3")
class BGEM3(VisionRetriever):
    def __init__(self, model: BGEM3FlagModel, processor: None, collator: VisionCollator, *args, **kwargs):
        self.model = model
        self.processor = processor
        self.collator = collator

    def to(self, device: str | torch.device) -> VisionRetriever:
        self.model = self.model.to(device)
        if self.processor:
            self.processor = self.processor.to(device)
        return self

    def forward_queries(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def forward_documents(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def embed_queries(self, queries: List[str]) -> torch.Tensor:
        return self.processor(queries)

    def embed_documents(self, documents: List[Image.Image]) -> torch.Tensor:
        return self.processor(documents)


@register_collator("BGEM3")
class CollatorBGEM3(VisionCollator):
    def __init__(self):
        pass

    def __call__(self, batch):
        return batch
