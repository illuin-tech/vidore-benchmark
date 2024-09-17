from __future__ import annotations

from typing import List, Optional, TypeVar, cast

import torch
from colpali_engine.models import ColPali
from dotenv import load_dotenv
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor

from vidore_benchmark.evaluation.colpali_scorer import ColPaliScorer
from vidore_benchmark.retrievers.utils.register_retriever import register_vision_retriever
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever
from vidore_benchmark.utils.torch_utils import get_torch_device

T = TypeVar("T")
load_dotenv(override=True)


class ListDataset(Dataset[T]):
    def __init__(self, elements: List[T]):
        self.elements = elements

    def __len__(self) -> int:
        return len(self.elements)

    def __getitem__(self, idx: int) -> T:
        return self.elements[idx]


@register_vision_retriever("vidore/colpali")
class ColPaliRetriever(VisionRetriever):
    """
    ColPali Retriever that implements the model from "ColPali: Efficient Document Retrieval
    with Vision Language Models".
    """

    def __init__(
        self,
        adapter_name: str = "vidore/colpali-v1.2",
        model_name: str = "vidore/colpaligemma-3b-pt-448-base",
        device: str = "auto",
    ):
        super().__init__()

        self.device = get_torch_device(device)
        logger.info(f"Using device: {self.device}")

        self.model = cast(
            ColPali,
            ColPali.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=device),
        ).eval()
        logger.info(f"Loaded ColPali model (non-trained weights) from `{model_name}`")

        self.model.load_adapter(adapter_name)
        logger.info(f"Loaded ColPali adapter from `{adapter_name}`")

        self.scorer = ColPaliScorer(is_multi_vector=True, device=self.device)
        self.processor = AutoProcessor.from_pretrained(adapter_name)
        self.emb_dim_query = 128
        self.emb_dim_doc = 128

    @property
    def use_visual_embedding(self) -> bool:
        return True

    def process_images(self, images: List[Image.Image], **kwargs):
        texts_doc = ["Describe the image."] * len(images)
        images = [image.convert("RGB") for image in images]

        batch_doc = self.processor(
            text=texts_doc,
            images=images,
            return_tensors="pt",
            padding="longest",
            max_length=kwargs.get("max_length", 50) + self.processor.image_seq_length,
        )
        return batch_doc

    def process_queries(self, queries: List[str], **kwargs) -> torch.Tensor:
        texts_query = []
        for query in queries:
            query = f"Question: {query}<unused0><unused0><unused0><unused0><unused0>"
            texts_query.append(query)

        mock_image = Image.new("RGB", (448, 448), (255, 255, 255))
        batch_query = self.processor(
            images=[mock_image.convert("RGB")] * len(texts_query),
            text=texts_query,
            return_tensors="pt",
            padding="longest",
            max_length=kwargs.get("max_length", 50) + self.processor.image_seq_length,
        )
        # NOTE: the image is not used in batch_query but it is required for calling the processor

        del batch_query["pixel_values"]

        batch_query["input_ids"] = batch_query["input_ids"][..., self.processor.image_seq_length :]
        batch_query["attention_mask"] = batch_query["attention_mask"][..., self.processor.image_seq_length :]
        return batch_query

    def forward_queries(self, queries: List[str], batch_size: int, **kwargs) -> List[torch.Tensor]:
        dataloader = DataLoader(
            dataset=ListDataset[str](queries),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.process_queries,
        )

        qs = []
        for batch_query in tqdm(dataloader, desc="Forward pass queries..."):
            with torch.no_grad():
                batch_query = {k: v.to(self.device) for k, v in batch_query.items()}
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

        ds = []
        for batch_doc in tqdm(dataloader, desc="Forward pass documents..."):
            with torch.no_grad():
                batch_doc = {k: v.to(self.device) for k, v in batch_doc.items()}
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
        scores = self.scorer.evaluate(list_emb_queries, list_emb_documents, batch_size)
        return scores
