from __future__ import annotations

import logging
from typing import ClassVar, List, Optional, cast

import torch
from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.utils.torch_utils import get_torch_device
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from vidore_benchmark.retrievers.utils.register_retriever import register_vision_retriever
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever
from vidore_benchmark.utils.data_utils import ListDataset

logger = logging.getLogger(__name__)

load_dotenv(override=True)


@register_vision_retriever("colpali")
class ColPaliRetriever(VisionRetriever):
    """
    ColPali retriever that implements the model from "ColPali: Efficient Document Retrieval
    with Vision Language Models".
    """

    emb_dim_query: ClassVar[int] = 128
    emb_dim_doc: ClassVar[int] = 128

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        device: str = "auto",
    ):
        super().__init__()

        self.device = get_torch_device(device)
        logger.info(f"Using device: {self.device}")

        # Load the model
        self.model = cast(
            ColPali,
            ColPali.from_pretrained(
                pretrained_model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
            ).eval(),
        )

        # Load the processor
        self.processor = cast(
            ColPaliProcessor,
            ColPaliProcessor.from_pretrained(pretrained_model_name_or_path),
        )

    @property
    def use_visual_embedding(self) -> bool:
        return True

    def process_images(self, images: List[Image.Image], **kwargs):
        return self.processor.process_images(images=images).to(self.device)

    def process_queries(self, queries: List[str], **kwargs):
        return self.processor.process_queries(queries=queries).to(self.device)

    def forward_queries(self, queries: List[str], batch_size: int, **kwargs) -> List[torch.Tensor]:
        dataloader = DataLoader(
            dataset=ListDataset[str](queries),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.process_queries,
        )

        qs: List[torch.Tensor] = []

        with torch.no_grad():
            for batch_query in tqdm(dataloader, desc="Forward pass queries...", leave=False):
                embeddings_query = self.model(**batch_query)
                qs.extend(list(torch.unbind(embeddings_query)))

        return qs

    def forward_documents(self, documents: List[Image.Image], batch_size: int, **kwargs) -> List[torch.Tensor]:
        dataloader = DataLoader(
            dataset=ListDataset[Image.Image](documents),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.process_images,
        )

        ds: List[torch.Tensor] = []

        with torch.no_grad():
            for batch_doc in tqdm(dataloader, desc="Forward pass documents...", leave=False):
                embeddings_doc = self.model(**batch_doc)
                ds.extend(list(torch.unbind(embeddings_doc)))
        return ds

    def get_scores(
        self,
        list_emb_queries: List[torch.Tensor],
        list_emb_documents: List[torch.Tensor],
        batch_size: Optional[int] = 128,
    ) -> torch.Tensor:
        if batch_size is None:
            raise ValueError("`batch_size` must be provided for ColPaliRetriever's scoring")
        scores = self.processor.score(
            list_emb_queries,
            list_emb_documents,
            batch_size=batch_size,
            device=self.device,
        )
        return scores
