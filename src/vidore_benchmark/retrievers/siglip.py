from typing import List

import torch
from PIL import Image
from torch._tensor import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from vidore_benchmark.retrievers.utils.register_models import register_vision_retriever
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever
from vidore_benchmark.utils.iter_utils import batched
from vidore_benchmark.utils.torch_utils import get_torch_device


@register_vision_retriever("google/siglip-so400m-patch14-384")
class SigLip(VisionRetriever):
    def __init__(self, visual_embedding: bool = True):
        super().__init__(visual_embedding)

        self.device = get_torch_device()
        self.processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        self.model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to(self.device)

    def forward_queries(self, queries: List[str], **kwargs) -> Tensor:
        inputs_queries = self.processor(text=queries, return_tensors="pt", padding="max_length", truncation=True).to(
            self.device
        )
        qs = self.model.get_text_features(**inputs_queries)

        return torch.tensor(qs).to(self.device)

    def forward_documents(self, documents: List[str | Image.Image], **kwargs) -> Tensor:
        input_image_processed = self.processor(images=documents, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            ps = self.model.get_image_features(**input_image_processed)

        return torch.tensor(ps).to(self.device)

    def get_scores(
        self, queries: List[str], documents: List[str | Image.Image], batch_query: int, batch_doc: int
    ) -> Tensor:
        list_emb_queries: List[torch.Tensor] = []
        for query_batch in tqdm(batched(queries, batch_query), desc="Query batch", total=len(queries) // batch_query):
            query_embeddings = self.forward_queries(query_batch)  # type: ignore
            list_emb_queries.append(query_embeddings)

        list_emb_documents: List[torch.Tensor] = []
        for doc_batch in tqdm(batched(documents, batch_doc), desc="Document batch", total=len(documents) // batch_doc):
            doc_embeddings = self.forward_documents(doc_batch)  # type: ignore
            list_emb_documents.append(doc_embeddings)

        emb_queries = torch.cat(list_emb_queries, dim=0)
        emb_documents = torch.cat(list_emb_documents, dim=0)

        scores = torch.einsum("bd,cd->bc", emb_queries, emb_documents)

        assert scores.shape == (emb_queries.shape[0], emb_documents.shape[0])

        return scores
