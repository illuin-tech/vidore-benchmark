import math
from typing import List, Union, cast

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel

from vidore_benchmark.retrievers.base_vision_retriever import BaseVisionRetriever
from vidore_benchmark.retrievers.registry_utils import register_vision_retriever
from vidore_benchmark.utils.iter_utils import batched
from vidore_benchmark.utils.torch_utils import get_torch_device

RETRIEVAL_QUERY_TASK = "retrieval.query"
RETRIEVAL_PASSAGE_TASK = "retrieval.passage"


@register_vision_retriever("jev3")
class JinaV3Retriever(BaseVisionRetriever):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "jinaai/jina-embeddings-v3",
        device: str = "auto",
        **kwargs,
    ):
        super().__init__(use_visual_embedding=False)
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


    def forward_queries(self, queries, batch_size: int, **kwargs) -> List[torch.Tensor]:
        list_emb_queries: List[torch.Tensor] = []
        for query_batch in tqdm(
            batched(queries, batch_size),
            desc="Forwarding query batches",
            total=math.ceil(len(queries) / batch_size),
            leave=False,
        ):
            query_batch = cast(List[str], query_batch)
            with torch.no_grad():
                query_embeddings = cast(
                    np.ndarray,
                    self.model.encode(
                        query_batch,
                        task=RETRIEVAL_QUERY_TASK,
                    ),
                )
                list_emb_queries.extend(query_embeddings.tolist())

        return torch.tensor(list_emb_queries)

    def forward_passages(self, passages: List[str], batch_size: int, **kwargs) -> List[torch.Tensor]:
        list_emb_passages: List[torch.Tensor] = []
        for passage_batch in tqdm(
            batched(passages, batch_size),
            desc="Forwarding passage batches",
            total=math.ceil(len(passages) / batch_size),
            leave=False,
        ):
            passage_batch = cast(List[str], passage_batch)
            with torch.no_grad():
                passage_embeddings = cast(
                    np.ndarray,
                    self.model.encode(
                        passage_batch,
                        task=RETRIEVAL_PASSAGE_TASK,
                    ),
                )
                list_emb_passages.extend(passage_embeddings.tolist())

        return torch.tensor(list_emb_passages)

    def get_scores(
        self,
        query_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        passage_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        **kwargs,
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
