import math
from typing import List, Optional, Union, cast

import numpy as np
import torch
from tqdm import tqdm

from vidore_benchmark.retrievers.registry_utils import register_vision_retriever
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever
from vidore_benchmark.utils.iter_utils import batched
from vidore_benchmark.utils.torch_utils import get_torch_device


@register_vision_retriever("bge-m3")
class BGEM3Retriever(VisionRetriever):
    """
    BGEM3Retriever class to retrieve embeddings the BGE-M3 model (dense embeddings).
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = "BAAI/bge-m3",
        device: str = "auto",
    ):
        super().__init__()

        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError:
            raise ImportError(
                'Install the missing dependencies with `pip install "vidore-benchmark[bge-m3]"` to use BGEM3Retriever.'
            )

        self.device = get_torch_device(device)

        self.model = BGEM3FlagModel(
            pretrained_model_name_or_path,
            use_fp16=True,
            device=self.device,
        )
        # NOTE: BGEM3FlagModel is already in eval mode

    @property
    def use_visual_embedding(self) -> bool:
        return False

    def forward_queries(self, queries, batch_size: int, **kwargs) -> torch.Tensor:
        list_emb_queries: List[float] = []

        for query_batch in tqdm(
            batched(queries, batch_size),
            desc="Forwarding query batches",
            total=math.ceil(len(queries) / batch_size),
            leave=False,
        ):
            query_batch = cast(List[str], query_batch)

            with torch.no_grad():
                query_embeddings = cast(np.ndarray, self.model.encode(query_batch, max_length=512)["dense_vecs"])

            list_emb_queries.extend(query_embeddings.tolist())

        return torch.tensor(list_emb_queries)

    def forward_passages(self, passages: List[str], batch_size: int, **kwargs) -> torch.Tensor:
        list_emb_passages: List[torch.Tensor] = []

        for passage_batch in tqdm(
            batched(passages, batch_size),
            desc="Forwarding passage batches",
            total=math.ceil(len(passages) / batch_size),
            leave=False,
        ):
            passage_batch = cast(List[str], passage_batch)

            with torch.no_grad():
                passage_embeddings = cast(np.ndarray, self.model.encode(passage_batch)["dense_vecs"])

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
