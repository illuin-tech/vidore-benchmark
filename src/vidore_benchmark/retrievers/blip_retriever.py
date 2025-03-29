from __future__ import annotations

import math
from typing import List, Optional, Union, cast

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from vidore_benchmark.retrievers.base_vision_retriever import BaseVisionRetriever
from vidore_benchmark.retrievers.registry_utils import register_vision_retriever
from vidore_benchmark.utils.iter_utils import batched
from vidore_benchmark.utils.torch_utils import get_torch_device


@register_vision_retriever("blip")
class BlipRetriever(BaseVisionRetriever):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "Salesforce/blip-itm-base-flickr",
        device: str = "auto",
        **kwargs,
    ):
        super().__init__(use_visual_embedding=True)

        try:
            import transformers  # noqa: F401
        except ImportError:
            raise ImportError(
                'Install the missing dependencies with `pip install "vidore-benchmark[transformers]"` '
                "to use BlipRetriever."
            )

        # blip models: https://huggingface.co/collections/Salesforce/blip-models-65242f40f1491fbf6a9e9472
        # blip2 models: https://huggingface.co/collections/Salesforce/blip2-models-65242f91b4c4b4a32e5cb652
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.device = get_torch_device(device)

        self.model = (
            AutoModel.from_pretrained(
                self.pretrained_model_name_or_path,
                trust_remote_code=True,
            )
            .to(self.device)
            .eval()
        )
        
        self.processor = AutoProcessor.from_pretrained(self.pretrained_model_name_or_path)

    def _encode_text(self, texts):
        # Process text inputs
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        # Get text features through the text model
        with torch.no_grad():
            text_outputs = self.model.text_model(**{k: v for k, v in inputs.items() if k in ["input_ids", "attention_mask"]})
            text_embeds = text_outputs[1]  # Pooled output
            text_embeds = self.model.text_projection(text_embeds)
            
            # Normalize embeddings
            text_embeds = F.normalize(text_embeds, dim=-1)
        
        return text_embeds.cpu().numpy()

    def _encode_image(self, images):
        # Process image inputs - handling both PIL images and file paths
        if isinstance(images[0], str):
            processed_images = [Image.open(img_path).convert("RGB") for img_path in images]
        else:
            processed_images = images
            
        inputs = self.processor(images=processed_images, return_tensors="pt").to(self.device)
        
        # Get image features through the vision model
        with torch.no_grad():
            vision_outputs = self.model.vision_model(inputs.pixel_values)
            image_embeds = vision_outputs[1]  # Pooled output
            image_embeds = self.model.visual_projection(image_embeds)
            
            # Normalize embeddings
            image_embeds = F.normalize(image_embeds, dim=-1)
        
        return image_embeds.cpu().numpy()

    def forward_queries(self, queries, batch_size: int, **kwargs) -> torch.Tensor:
        list_emb_queries: List[torch.Tensor] = []
        for query_batch in tqdm(
            batched(queries, batch_size),
            desc="Forwarding query batches",
            total=math.ceil(len(queries) / batch_size),
            leave=False,
        ):
            query_batch = cast(List[str], query_batch)
            with torch.no_grad():
                query_embeddings = self._encode_text(query_batch)
                list_emb_queries.extend(query_embeddings.tolist())

        return torch.tensor(list_emb_queries)

    def forward_passages(self, passages, batch_size: int, **kwargs) -> torch.Tensor:
        list_emb_passages: List[torch.Tensor] = []
        for passage_batch in tqdm(
            batched(passages, batch_size),
            desc="Forwarding passage batches",
            total=math.ceil(len(passages) / batch_size),
            leave=False,
        ):
            passage_batch = cast(List[str], passage_batch)
            with torch.no_grad():
                passage_embeddings = self._encode_image(passage_batch)
                list_emb_passages.extend(passage_embeddings.tolist())

        return torch.tensor(list_emb_passages)

    def get_scores(
        self,
        query_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        passage_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Dot-product similarity between queries and passages.
        """
        if isinstance(query_embeddings, list):
            query_embeddings = torch.stack(query_embeddings)
        if isinstance(passage_embeddings, list):
            passage_embeddings = torch.stack(passage_embeddings)

        scores = torch.einsum("bd,cd->bc", query_embeddings, passage_embeddings)
        return scores
