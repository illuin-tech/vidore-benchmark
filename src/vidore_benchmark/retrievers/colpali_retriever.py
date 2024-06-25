from __future__ import annotations

from abc import abstractmethod
from typing import List, cast

import torch
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor

from vidore_benchmark.evaluation.colpali_scorer import ColPaliScorer
from vidore_benchmark.models.colpali_model import ColPali
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever
from vidore_benchmark.utils.torch_utils import get_torch_device
from vidore_benchmark.retrievers.utils.register_models import register_vision_retriever

load_dotenv(override=True)


@register_vision_retriever("coldoc/colpali-3b-mix-448")
class ColPaliRetriever(VisionRetriever):
    """
    ColPali Retriever that implements the model from "ColPali: Efficient Document Retrieval with Vision Language Models".
    """

    def __init__(self, device: str = "auto"):
        super().__init__()
        model_name = "coldoc/colpali-3b-mix-448"
        self.device = get_torch_device(device)
        self.model = cast(
            ColPali,
            ColPali.from_pretrained("google/paligemma-3b-mix-448", torch_dtype=torch.bfloat16, device_map=device),
        ).eval()
        self.model.load_adapter(model_name)
        self.scorer = ColPaliScorer(is_multi_vector=True)
        self.processor = AutoProcessor.from_pretrained(model_name)

    @property
    def use_visual_embedding(self) -> bool:
        return True

    def process_images(self, images, **kwargs):
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

    def process_queries(self, queries, **kwargs) -> torch.Tensor:
        texts_query = []
        for query in queries:
            query = f"Question: {query}<unused0><unused0><unused0><unused0><unused0>"
            texts_query.append(query)

        mock_image = Image.new("RGB", (448, 448), (255, 255, 255))
        batch_query = self.processor(
            images=[mock_image.convert("RGB")] * len(texts_query),
            # NOTE: the image is not used in batch_query but it is required for calling the processor
            text=texts_query,
            return_tensors="pt",
            padding="longest",
            max_length=kwargs.get("max_length", 50) + self.processor.image_seq_length,
        )
        del batch_query["pixel_values"]

        batch_query["input_ids"] = batch_query["input_ids"][..., self.processor.image_seq_length :]
        batch_query["attention_mask"] = batch_query["attention_mask"][..., self.processor.image_seq_length :]
        return batch_query

    def forward_queries(self, queries, **kwargs) -> List[torch.Tensor]:
        """
        Forward pass the processed queries.
        """
        # run inference - docs
        dataloader = DataLoader(
            queries,
            batch_size=kwargs.get("bs", 4),
            shuffle=False,
            collate_fn=self.process_queries,
        )
        qs = []
        for batch_query in tqdm(dataloader):
            with torch.no_grad():
                batch_query = {k: v.to(self.device) for k, v in batch_query.items()}
                embeddings_query = self.model(**batch_query)
                qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))

        return qs

    def forward_documents(self, documents, **kwargs) -> List[torch.Tensor]:
        """
        Forward pass the processed documents (i.e. page images).
        """

        # run inference - docs
        dataloader = DataLoader(
            documents,
            batch_size=kwargs.get("bs", 4),
            shuffle=False,
            collate_fn=lambda x: self.process_images(x),
        )
        ds = []
        for batch_doc in tqdm(dataloader):
            with torch.no_grad():
                batch_doc = {k: v.to(self.device) for k, v in batch_doc.items()}
                embeddings_doc = self.model(**batch_doc)
            ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))
        return ds

    def get_scores(
        self,
        queries: List[str],
        documents: List[Image.Image | str],
        batch_query: int,
        batch_doc: int,
        **kwargs,
    ) -> torch.Tensor:
        """
        Get the similarity scores between queries and documents.
        """
        qs = self.forward_queries(queries, bs=batch_query)
        ds = self.forward_documents(documents, bs=batch_doc)

        # Unpack the stacked embeddings to a list for the scorer
        # qs = list(torch.unbind(qs_stacked))
        # ds = list(torch.unbind(ds_stacked))

        scores = self.scorer.evaluate(qs, ds)
        return scores
