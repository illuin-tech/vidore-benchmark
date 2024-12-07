from __future__ import annotations

import math
from typing import List, Optional, Union, cast

import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image
from torch import Tensor
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

from vidore_benchmark.retrievers.registry_utils import register_vision_retriever
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever
from vidore_benchmark.utils.iter_utils import batched
from vidore_benchmark.utils.torch_utils import get_torch_device


@register_vision_retriever("nomic-embed-vision")
class NomicVisionRetriever(VisionRetriever):
    def __init__(self, device: str = "auto"):
        super().__init__()
        self.device = get_torch_device(device)

        self.model = (
            AutoModel.from_pretrained(
                "nomic-ai/nomic-embed-vision-v1.5",
                trust_remote_code=True,
            )
            .to(self.device)
            .eval()
        )
        self.processor = AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5")

        self.text_model = AutoModel.from_pretrained(
            "nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True,
        ).to(self.device)

        self.text_tokenizer = AutoTokenizer.from_pretrained(
            "nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True,
        )

    @property
    def use_visual_embedding(self) -> bool:
        return True

    @staticmethod
    def _mean_pooling(model_output: Tensor, attention_mask: Tensor) -> Tensor:
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward_queries(self, queries, batch_size: int, **kwargs) -> List[torch.Tensor]:
        list_emb_queries: List[torch.Tensor] = []
        for query_batch in tqdm(
            batched(queries, batch_size),
            desc="Forwarding query batches",
            total=math.ceil(len(queries) / batch_size),
            leave=False,
        ):
            query_batch = cast(List[str], query_batch)

            query_texts = ["search_query: " + query for query in query_batch]
            encoded_input = self.text_tokenizer(
                query_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                qs = self.text_model(**encoded_input)
                qs = self._mean_pooling(qs, encoded_input["attention_mask"])
                qs = F.layer_norm(qs, normalized_shape=(qs.shape[1],))
                qs = F.normalize(qs, p=2, dim=1)

            query_embeddings = torch.tensor(qs).to(self.device)
            list_emb_queries.extend(list(torch.unbind(query_embeddings, dim=0)))

        return list_emb_queries

    def forward_passages(self, passages, batch_size: int, **kwargs) -> List[torch.Tensor]:
        list_emb_passages: List[torch.Tensor] = []
        for passage_batch in tqdm(
            batched(passages, batch_size),
            desc="Forwarding passage batches",
            total=math.ceil(len(passages) / batch_size),
            leave=False,
        ):
            passage_batch = cast(List[Image.Image], passage_batch)

            vision_inputs = self.processor(passage_batch, return_tensors="pt").to(self.device)
            with torch.no_grad():
                ps = self.model(**vision_inputs).last_hidden_state
                ps = F.normalize(ps[:, 0], p=2, dim=1)

            passage_embeddings = torch.tensor(ps).to(self.device)
            list_emb_passages.extend(list(torch.unbind(passage_embeddings, dim=0)))

        return list_emb_passages

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
