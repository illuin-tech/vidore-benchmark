import math
from typing import List, Optional, cast

import torch
from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm

from vidore_benchmark.retrievers.utils.register_retriever import register_vision_retriever
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever
from vidore_benchmark.utils.iter_utils import batched
from vidore_benchmark.utils.torch_utils import get_torch_device


@register_vision_retriever("BAAI/bge-m3")
class BGEM3Retriever(VisionRetriever):
    """
    BGEM3Retriever class to retrieve embeddings the BGE-M3 model (dense embeddings).
    """

    def __init__(self, device: str = "auto"):
        super().__init__()
        self.device = get_torch_device(device)
        self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        self.emb_dim_query = 1024
        self.emb_dim_doc = 1024

    @property
    def use_visual_embedding(self) -> bool:
        return False

    def forward_queries(self, queries, batch_size: int, **kwargs) -> List[torch.Tensor]:
        list_emb_queries: List[torch.Tensor] = []
        for query_batch in tqdm(
            batched(queries, batch_size), desc="Query batch", total=math.ceil(len(queries) / batch_size)
        ):
            query_batch = cast(List[str], query_batch)
            with torch.no_grad():
                output = self.model.encode(query_batch, max_length=512)["dense_vecs"]
            query_embeddings = torch.tensor(output).to(self.device)
            list_emb_queries.append(query_embeddings)

        return list_emb_queries

    def forward_documents(self, documents: List[str], batch_size: int, **kwargs) -> List[torch.Tensor]:
        list_emb_documents: List[torch.Tensor] = []
        for doc_batch in tqdm(
            batched(documents, batch_size), desc="Document batch", total=math.ceil(len(documents) / batch_size)
        ):
            doc_batch = cast(List[str], doc_batch)
            with torch.no_grad():
                output = self.model.encode(doc_batch)["dense_vecs"]
            doc_embeddings = torch.tensor(output).to(self.device)
            list_emb_documents.append(doc_embeddings)
        return list_emb_documents

    def get_scores(
        self,
        list_emb_queries: List[torch.Tensor],
        list_emb_documents: List[torch.Tensor],
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        emb_queries = torch.cat(list_emb_queries, dim=0)  # (num_queries, emb_dim_query)
        emb_documents = torch.cat(list_emb_documents, dim=0)  # (num_docs, emb_dim_doc)

        scores = torch.einsum("bd,cd->bc", emb_queries, emb_documents)  # (num_queries, num_docs)
        return scores
