import math
from typing import List, Optional, cast

import torch
from colpali_engine.utils.torch_utils import get_torch_device
from tqdm import tqdm

from vidore_benchmark.evaluation.colbert_score import get_colbert_similarity
from vidore_benchmark.retrievers.utils.register_retriever import register_vision_retriever
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever
from vidore_benchmark.utils.iter_utils import batched

try:
    from FlagEmbedding import BGEM3FlagModel
except ImportError:
    pass


@register_vision_retriever("bge-m3-colbert")
class BGEM3ColbertRetriever(VisionRetriever):
    """
    BGEM3Retriever class to retrieve embeddings the BGE-M3 model (multi-vector embeddings + ColBERT scoring).
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = "BAAI/bge-m3",
        device: str = "auto",
    ):
        super().__init__()
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

    def forward_queries(self, queries: List[str], batch_size: int, **kwargs) -> List[torch.Tensor]:
        list_emb_queries: List[torch.Tensor] = []
        for query_batch in tqdm(
            batched(queries, batch_size),
            desc="Query batch",
            total=math.ceil(len(queries) / batch_size),
            leave=False,
        ):
            query_batch = cast(List[str], query_batch)
            with torch.no_grad():
                output = self.model.encode(
                    query_batch,
                    max_length=512,
                    return_dense=False,
                    return_sparse=False,
                    return_colbert_vecs=True,
                )["colbert_vecs"]
            list_emb_queries.extend([torch.Tensor(elt) for elt in output])
        return list_emb_queries

    def forward_documents(self, documents: List[str], batch_size: int, **kwargs) -> List[torch.Tensor]:
        list_emb_documents: List[torch.Tensor] = []
        for doc_batch in tqdm(
            batched(documents, batch_size),
            desc="Document batch",
            total=math.ceil(len(documents) / batch_size),
            leave=False,
        ):
            doc_batch = cast(List[str], doc_batch)
            with torch.no_grad():
                output = self.model.encode(
                    doc_batch,
                    max_length=512,
                    return_dense=False,
                    return_sparse=False,
                    return_colbert_vecs=True,
                )["colbert_vecs"]
            list_emb_documents.extend([torch.Tensor(elt) for elt in output])
        return list_emb_documents

    def get_scores(
        self,
        list_emb_queries: List[torch.Tensor],
        list_emb_documents: List[torch.Tensor],
        batch_size: Optional[int] = 4,
    ) -> torch.Tensor:
        if batch_size is None:
            raise ValueError("The batch size must be specified for the ColBERT scoring.")
        scores = get_colbert_similarity(list_emb_queries, list_emb_documents, batch_size=batch_size)
        return scores
