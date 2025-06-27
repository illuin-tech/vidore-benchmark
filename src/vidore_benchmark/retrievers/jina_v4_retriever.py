from __future__ import annotations

import logging
from typing import List, Optional, Union

import torch
from dotenv import load_dotenv
from PIL import Image
from transformers import AutoModel

from vidore_benchmark.retrievers.base_vision_retriever import BaseVisionRetriever
from vidore_benchmark.retrievers.registry_utils import register_vision_retriever
from vidore_benchmark.utils.torch_utils import get_torch_device

logger = logging.getLogger(__name__)

load_dotenv(override=True)

@register_vision_retriever("jev4")
class JinaV4Retriever(BaseVisionRetriever):
    def __init__(
        self,
        pretrained_model_name_or_path: Optional[str] = "jinaai/jina-embeddings-v4",
        device: str = "auto",
        show_progress: bool = True,
        vector_type: Optional[str] = "single_vector",
        max_length: Optional[int] = None,
        truncate: Optional[int] = None,
        max_pixels: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(use_visual_embedding=True)

        self.device = get_torch_device(device)
        self.show_progress = show_progress
        self.vector_type = vector_type
        self.max_length = max_length
        self.truncate = truncate
        self.max_pixels = max_pixels
        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        self.model = self.model.eval().to(self.device)
        self.model.task = 'retrieval'

    def forward_queries(self, queries: List[str], batch_size: int, **kwargs) -> List[torch.Tensor]:
        return self.model.encode_text(
            texts=queries,
            max_length=self.max_length,
            truncate_dim=self.truncate,
            batch_size=batch_size,
            return_multivector = (self.vector_type.lower() == 'multi_vector'),
            **kwargs,
        )

    def forward_passages(self, passages: List[Image.Image], batch_size: int, **kwargs) -> List[torch.Tensor]:
        return self.model.encode_image(
            images=passages,
            batch_size=batch_size,
            truncate_dim=self.truncate,
            max_pixels=self.max_pixels,
            return_multivector = (self.vector_type.lower() == 'multi_vector'),
            **kwargs,
        )

    def get_scores(
        self,
        query_embeddings: List[torch.Tensor],
        passage_embeddings: List[torch.Tensor],
        batch_size: Optional[int] = 128,
    ) -> torch.Tensor:
        if batch_size is None:
            raise ValueError("`batch_size` must be provided for ColPaliRetriever's scoring")
        if self.vector_type == "single_vector":
            return self.score_single_vector(query_embeddings, passage_embeddings, device=self.device)
        elif self.vector_type == "multi_vector":
            return self.score_multi_vector(query_embeddings, passage_embeddings, device=self.device)
        else:
            raise ValueError('vector_type must be one of the following: [`single_vector`, `multi_vector`]')

    @staticmethod
    def score_single_vector(
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Compute the dot product score for the given single-vector query and passage embeddings.
        """
        device = device or get_torch_device("auto")

        if len(qs) == 0:
            raise ValueError("No queries provided")
        if len(ps) == 0:
            raise ValueError("No passages provided")

        qs_stacked = torch.stack(qs).to(device)
        ps_stacked = torch.stack(ps).to(device)

        scores = torch.einsum("bd,cd->bc", qs_stacked, ps_stacked)
        assert scores.shape[0] == len(qs), f"Expected {len(qs)} scores, got {scores.shape[0]}"

        scores = scores.to(torch.float32)
        return scores

    @staticmethod
    def score_multi_vector(
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        batch_size: int = 16,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Compute the MaxSim score (ColBERT-like) for the given multi-vector query and passage embeddings.
        """
        device = device or get_torch_device("auto")

        if len(qs) == 0:
            raise ValueError("No queries provided")
        if len(ps) == 0:
            raise ValueError("No passages provided")

        scores_list: List[torch.Tensor] = []

        for i in range(0, len(qs), batch_size):
            scores_batch = []
            qs_batch = torch.nn.utils.rnn.pad_sequence(qs[i: i + batch_size], batch_first=True, padding_value=0).to(
                device
            )
            for j in range(0, len(ps), batch_size):
                ps_batch = torch.nn.utils.rnn.pad_sequence(
                    ps[j: j + batch_size], batch_first=True, padding_value=0
                ).to(device)
                scores_batch.append(torch.einsum("bnd,csd->bcns", qs_batch, ps_batch).max(dim=3)[0].sum(dim=2))
            scores_batch = torch.cat(scores_batch, dim=1).cpu()
            scores_list.append(scores_batch)

        scores = torch.cat(scores_list, dim=0)
        assert scores.shape[0] == len(qs), f"Expected {len(qs)} scores, got {scores.shape[0]}"

        scores = scores.to(torch.float32)
        return scores
