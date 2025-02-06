from __future__ import annotations

import logging
from typing import List, Optional, Union

import torch
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ProcessorMixin

from vidore_benchmark.retrievers.base_vision_retriever import BaseVisionRetriever
from vidore_benchmark.utils.data_utils import ListDataset

logger = logging.getLogger(__name__)

load_dotenv(override=True)


class VisionRetriever(BaseVisionRetriever):
    """
    Vision Retriever wrapper class that can be used with `BaseViDoReEvaluator`.

    To use this class, the following requirements must be met:
    - `model` has a `forward` method that returns dense or multi-vector embeddings
    - `processor` implements `process_images`, `process_queries`, and `score` methods.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        processor: ProcessorMixin,
    ):
        super().__init__(use_visual_embedding=True)

        self.model = model
        self.model.eval()

        self.processor = processor
        if not hasattr(self.processor, "process_images"):
            raise ValueError("Processor must have `process_images` method")
        if not hasattr(self.processor, "process_queries"):
            raise ValueError("Processor must have `process_queries` method")
        if not hasattr(self.processor, "score"):
            raise ValueError("Processor must have `score` method")

    def process_images(self, images: List[Image.Image], **kwargs):
        return self.processor.process_images(images).to(self.model.device)

    def process_queries(self, queries: List[str], **kwargs):
        return self.processor.process_queries(queries).to(self.model.device)

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
        )

        query_embeddings: List[torch.Tensor] = []

        with torch.no_grad():
            for batch_query in tqdm(dataloader, desc="Forward pass queries...", leave=False):
                embeddings_query = self.model(**batch_query).to("cpu")
                query_embeddings.extend(list(torch.unbind(embeddings_query)))

        return query_embeddings

    def forward_passages(self, passages: List[Image.Image], batch_size: int, **kwargs) -> List[torch.Tensor]:
        dataloader = DataLoader(
            dataset=ListDataset[Image.Image](passages),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.process_images,
        )

        passage_embeddings: List[torch.Tensor] = []

        with torch.no_grad():
            for batch_doc in tqdm(dataloader, desc="Forward pass passages...", leave=False):
                embeddings_doc = self.model(**batch_doc).to("cpu")
                passage_embeddings.extend(list(torch.unbind(embeddings_doc)))

        return passage_embeddings

    def get_scores(
        self,
        query_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        passage_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: Optional[int] = 128,
    ) -> torch.Tensor:
        if batch_size is None:
            raise ValueError("`batch_size` must be provided for ColPaliRetriever's scoring")
        scores = self.processor.score(
            qs=query_embeddings,
            ps=passage_embeddings,
            batch_size=batch_size,
            device="cpu",
        )
        return scores
