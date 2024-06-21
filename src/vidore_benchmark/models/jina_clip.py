from vidore_benchmark.models.vision_retriever import VisionRetriever
import torch
from PIL import Image
from typing import List, Dict, Any
from vidore_benchmark.models.utils.register_models import register_collator, register_vision_retriever
from vidore_benchmark.dataset.vision_collator import VisionCollator
from vidore_benchmark.utils.torch_utils import get_torch_device


@register_vision_retriever("jinaai/jina-clip-v1")
class JinaClip(VisionRetriever):
    def __init__(self, model: torch.nn.Module, processor: None, collator: VisionCollator, *args, **kwargs):

        self.device = get_torch_device()

        self.model = model.to(self.device)
        self.processor = processor
        self.collator = collator
        self.is_vision_retriever = True
        self.is_multi_vector = False

    def forward_queries(self, queries: List[str], **kwargs) -> torch.Tensor:
        output = self.model.encode_text(queries)
        print(output.shape)
        return torch.tensor(output).to(self.device)

    def forward_documents(self, documents: List[Image.Image], **kwargs) -> torch.Tensor:
        output = self.model.encode_image(documents)
        return torch.tensor(output).to(self.device)


@register_collator("jinaai/jina-clip-v1")
class CollatorJinaClip(VisionCollator):
    def __init__(self):
        self.col_document: str = "image"
        self.col_query: str = "query"

    def __call__(self, batch: Dict[str, List[Dict[str, torch.Tensor]]]) -> Any:
        documents = [item[self.col_document] for item in batch]
        return {"document": documents}
