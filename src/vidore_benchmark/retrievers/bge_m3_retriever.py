import math
from typing import List, Optional, Union, cast

import torch
from colpali_engine.utils.torch_utils import get_torch_device
from tqdm import tqdm

from vidore_benchmark.retrievers.registry_utils import register_vision_retriever
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever
from vidore_benchmark.utils.iter_utils import batched


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
            raise ImportError("Please install the `FlagEmbedding` package to use BGEM3Retriever.")

        self.device = get_torch_device(device)

        self.model = BGEM3FlagModel(
            pretrained_model_name_or_path,
            use_fp16=True,
            device=self.device,
        )
        # NOTE: BGEM3FlagModel is already in eval mode

        self.emb_dim_query = 1024
        self.emb_dim_doc = 1024

    @property
    def use_visual_embedding(self) -> bool:
        return False

    def forward_queries(self, queries, batch_size: int, **kwargs) -> List[torch.Tensor]:
        list_emb_queries: List[torch.Tensor] = []
        for query_batch in tqdm(
            batched(queries, batch_size),
            desc="Query batch",
            total=math.ceil(len(queries) / batch_size),
            leave=False,
        ):
            query_batch = cast(List[str], query_batch)
            with torch.no_grad():
                output = self.model.encode(query_batch, max_length=512)["dense_vecs"]
            query_embeddings = torch.tensor(output).to(self.device)
            list_emb_queries.append(query_embeddings)

        return list_emb_queries

    def forward_passages(self, passages: List[str], batch_size: int, **kwargs) -> List[torch.Tensor]:
        list_emb_passages: List[torch.Tensor] = []
        for doc_batch in tqdm(
            batched(passages, batch_size),
            desc="Document batch",
            total=math.ceil(len(passages) / batch_size),
            leave=False,
        ):
            doc_batch = cast(List[str], doc_batch)
            with torch.no_grad():
                output = self.model.encode(doc_batch)["dense_vecs"]
            doc_embeddings = torch.tensor(output).to(self.device)
            list_emb_passages.append(doc_embeddings)
        return list_emb_passages

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
