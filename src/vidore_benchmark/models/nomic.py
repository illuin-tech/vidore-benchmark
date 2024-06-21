from vidore_benchmark.models.vision_retriever import VisionRetriever
import torch
from PIL import Image
from typing import List, Dict, Any
from vidore_benchmark.models.utils.register_models import register_collator, register_vision_retriever
from vidore_benchmark.dataset.vision_collator import VisionCollator
from vidore_benchmark.utils.torch_utils import get_torch_device
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
from torch import Tensor
import torch.nn.functional as F

def mean_pooling(model_output: Tensor, attention_mask: Tensor) -> Tensor:
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

@register_vision_retriever("nomic-ai/nomic-embed-vision-v1.5")
class NomicVision(VisionRetriever):
    def __init__(
        self, model: torch.nn.Module, processor: AutoImageProcessor, collator: VisionCollator, *args, **kwargs
    ):

        self.device = get_torch_device()

        self.model = model.to(self.device)
        self.processor = processor
        self.collator = collator

        # for nomic only
        self.text_model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True).to(
            self.device
        )
        self.text_tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

        self.is_vision_retriever = True
        self.is_multi_vector = False

    def to(self, device: str | torch.device) -> VisionRetriever:
        self.model.to(device)
        return self

    def forward_queries(self, queries: List[str], **kwargs) -> torch.Tensor:
        query_texts = ["search_query: " + query for query in queries]
        encoded_input = self.text_tokenizer(query_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            qs = self.text_model(**encoded_input)
        qs = mean_pooling(qs, encoded_input["attention_mask"])
        qs = F.layer_norm(qs, normalized_shape=(qs.shape[1],))
        qs = F.normalize(qs, p=2, dim=1)

        return torch.tensor(qs).to(self.device)

    def forward_documents(self, documents: List[Image.Image], **kwargs) -> torch.Tensor:
        vision_inputs = self.processor(documents, return_tensors="pt").to(self.device)
        with torch.no_grad():
            ps = self.model(**vision_inputs).last_hidden_state
            ps = F.normalize(ps[:, 0], p=2, dim=1)

        return torch.tensor(ps).to(self.device)


@register_collator("nomic-ai/nomic-embed-vision-v1.5")
class CollatorJinaClip(VisionCollator):
    def __init__(self):
        self.col_document: str = "image"
        self.col_query: str = "query"

    def __call__(self, batch: Dict[str, List[Dict[str, torch.Tensor]]]) -> Any:
        documents = [item[self.col_document] for item in batch]
        return {"document": documents}
