from typing import List, cast

import torch
from FlagEmbedding import BGEM3FlagModel
from PIL import Image
from tqdm import tqdm

from vidore_benchmark.retrievers.utils.register_retriever import register_vision_retriever
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever
from vidore_benchmark.utils.iter_utils import batched
from vidore_benchmark.utils.torch_utils import get_torch_device


@register_vision_retriever("BAAI/bge-m3")
class BGEM3Retriever(VisionRetriever):
    def __init__(self, device: str = "auto"):
        super().__init__()
        self.device = get_torch_device(device)
        self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        self.emb_dim_query = 1024
        self.emb_dim_doc = 1024

    @property
    def use_visual_embedding(self) -> bool:
        return False

    def forward_queries(self, queries, **kwargs) -> torch.Tensor:
        output = self.model.encode(queries, max_length=512)["dense_vecs"]
        return torch.tensor(output).to(self.device)

    def forward_documents(self, documents: List[str], **kwargs) -> torch.Tensor:
        output = self.model.encode(documents)["dense_vecs"]
        return torch.tensor(output).to(self.device)

    def get_scores(
        self,
        queries: List[str],
        documents: List[str] | List[Image.Image],
        batch_query: int,
        batch_doc: int,
        **kwargs,
    ) -> torch.Tensor:

        # Sanity check: `documents` must be a list of filepaths (strings)
        if documents and not all(isinstance(doc, str) for doc in documents):
            raise ValueError("Documents must be a list of filepaths (strings)")
        documents = cast(List[str], documents)

        list_emb_queries: List[torch.Tensor] = []
        for query_batch in tqdm(batched(queries, batch_query), desc="Query batch", total=len(queries) // batch_query):
            query_batch = cast(List[str], query_batch)
            query_embeddings = self.forward_queries(query_batch)
            list_emb_queries.append(query_embeddings)

        list_emb_documents: List[torch.Tensor] = []
        for doc_batch in tqdm(batched(documents, batch_doc), desc="Document batch", total=len(documents) // batch_doc):
            doc_batch = cast(List[str], doc_batch)
            doc_embeddings = self.forward_documents(doc_batch)
            list_emb_documents.append(doc_embeddings)

        emb_queries = torch.cat(list_emb_queries, dim=0)
        emb_documents = torch.cat(list_emb_documents, dim=0)

        scores = torch.einsum("bd,cd->bc", emb_queries, emb_documents)

        return scores
