import math
from typing import List, Optional, Union, cast

import torch
from colpali_engine.utils.torch_utils import get_torch_device
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from vidore_benchmark.retrievers.registry_utils import register_vision_retriever
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever
from vidore_benchmark.utils.iter_utils import batched


@register_vision_retriever("siglip")
class SigLIPRetriever(VisionRetriever):
    """
    SigLIPRetriever class to retrieve embeddings from the SigLIP model.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = "google/siglip-so400m-patch14-384",
        device: str = "auto",
    ):
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.device = get_torch_device(device)

        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path)

    @property
    def use_visual_embedding(self) -> bool:
        return True

    def forward_queries(self, queries, batch_size: int, **kwargs) -> List[torch.Tensor]:
        list_emb_queries: List[torch.Tensor] = []
        for query_batch in tqdm(
            batched(queries, batch_size),
            desc="Query batch",
            total=math.ceil(len(queries) / batch_size),
            leave=False,
        ):
            query_batch = cast(List[str], query_batch)
            inputs_queries = self.processor(
                text=query_batch, return_tensors="pt", padding="max_length", truncation=True
            ).to(self.device)
            qs = self.model.get_text_features(**inputs_queries)
            query_embeddings = torch.tensor(qs).to(self.device)
            list_emb_queries.append(query_embeddings)
        return list_emb_queries

    def forward_documents(self, documents, batch_size: int, **kwargs) -> List[torch.Tensor]:
        list_emb_documents: List[torch.Tensor] = []
        for doc_batch in tqdm(
            batched(documents, batch_size),
            desc="Document batch",
            total=math.ceil(len(documents) / batch_size),
            leave=False,
        ):
            doc_batch = cast(List[Image.Image], doc_batch)
            list_doc = [document.convert("RGB") for document in doc_batch if isinstance(document, Image.Image)]

            input_image_processed = self.processor(images=list_doc, return_tensors="pt", padding=True).to(self.device)

            with torch.no_grad():
                ps = self.model.get_image_features(**input_image_processed)
                doc_embeddings = torch.tensor(ps).to(self.device)
                list_emb_documents.append(doc_embeddings)

        return list_emb_documents

    def get_scores(
        self,
        query_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        passage_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        if isinstance(query_embeddings, list):
            query_embeddings = torch.cat(query_embeddings, dim=0)
        if isinstance(passage_embeddings, list):
            passage_embeddings = torch.cat(passage_embeddings, dim=0)

        scores = torch.einsum("bd,cd->bc", query_embeddings, passage_embeddings)
        return scores
