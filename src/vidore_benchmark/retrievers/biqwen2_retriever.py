from __future__ import annotations

import logging
import os
from typing import List, Optional, Union, cast

import torch
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.utils.import_utils import is_flash_attn_2_available

from vidore_benchmark.retrievers.base_vision_retriever import BaseVisionRetriever
from vidore_benchmark.retrievers.registry_utils import register_vision_retriever
from vidore_benchmark.utils.data_utils import ListDataset
from vidore_benchmark.utils.torch_utils import get_torch_device

logger = logging.getLogger(__name__)

load_dotenv(override=True)


@register_vision_retriever("biqwen2")
class BiQwen2Retriever(BaseVisionRetriever):
    """
    BiQwen2 retriever that implements the model from "ColPali: Efficient Document Retrieval
    with Vision Language Models".
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        device: str = "auto",
        num_workers: Optional[int] = None,
    ):
        super().__init__(use_visual_embedding=True)

        try:
            from colpali_engine.models import BiQwen2, BiQwen2Processor
        except ImportError:
            raise ImportError(
                'Install the missing dependencies with `pip install "vidore-benchmark[colpali-engine]"` '
                "to use BiQwen2Retriever."
            )

        self.device = get_torch_device(device)
        logger.info(f"Using device: {self.device}")

        # Load the model and LORA adapter
        self.model = cast(
            BiQwen2,
            BiQwen2.from_pretrained(
                pretrained_model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map=device,
                attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
            ).eval(),
        )

        # Load the processor
        self.processor = cast(BiQwen2Processor, BiQwen2Processor.from_pretrained(pretrained_model_name_or_path))

        if num_workers is None:
            if self.device == "mps":
                self.num_workers = 0  # MPS does not support dataloader multiprocessing
            else:
                self.num_workers = os.cpu_count() if os.cpu_count() is not None else 1
        else:
            self.num_workers = num_workers

    def process_images(self, images: List[Image.Image], **kwargs):
        return self.processor.process_images(images=images).to(self.device)

    def process_queries(self, queries: List[str], **kwargs):
        return self.processor.process_queries(queries=queries).to(self.device)

    def forward_queries(
        self,
        queries: List[str],
        batch_size: int,
        **kwargs,
    ) -> List[torch.Tensor]:
        dataloader = DataLoader(
            dataset=ListDataset[str](queries),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.process_queries,
            num_workers=self.num_workers,
        )

        query_embeddings: List[torch.Tensor] = []

        for batch_query in tqdm(dataloader, desc="Forward pass queries...", leave=False):
            with torch.no_grad():
                batch_query = {k: v.to(self.device) for k, v in batch_query.items()}
                embeddings_query = self.model(**batch_query)
                query_embeddings.extend(list(torch.unbind(embeddings_query.to("cpu"))))

        return query_embeddings

    def forward_passages(
        self,
        passages: List[Image.Image],
        batch_size: int,
        **kwargs,
    ) -> List[torch.Tensor]:
        dataloader = DataLoader(
            dataset=ListDataset[Image.Image](passages),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.process_images,
            num_workers=self.num_workers,
        )

        passage_embeddings: List[torch.Tensor] = []

        for batch_doc in tqdm(dataloader, desc="Forward pass documents...", leave=False):
            with torch.no_grad():
                batch_doc = {k: v.to(self.device) for k, v in batch_doc.items()}
                embeddings_doc = self.model(**batch_doc)
            passage_embeddings.extend(list(torch.unbind(embeddings_doc.to("cpu"))))

        return passage_embeddings

    def get_scores(
        self,
        query_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        passage_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: Optional[int] = 128,
    ) -> torch.Tensor:
        if batch_size is None:
            raise ValueError("`batch_size` must be provided for BiQwenRetriever's scoring")
        scores = self.processor.score(
            query_embeddings,
            passage_embeddings,
            batch_size=batch_size,
            device="cpu",
        )
        return scores
