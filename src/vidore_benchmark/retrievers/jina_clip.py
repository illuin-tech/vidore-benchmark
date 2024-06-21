from vidore_benchmark.retrievers.vision_retriever import VisionRetriever
import torch
from PIL import Image
from typing import List
from vidore_benchmark.retrievers.utils.register_models import register_vision_retriever
from vidore_benchmark.utils.torch_utils import get_torch_device
from vidore_benchmark.utils.iter_utils import batched
from transformers import AutoModel
from tqdm import tqdm


@register_vision_retriever("jinaai/jina-clip-v1")
class JinaClip(VisionRetriever):
    def __init__(self, *args, **kwargs):

        self.device = get_torch_device()
        self.model = AutoModel.from_pretrained("jinaai/jina-clip-v1", trust_remote_code=True).to(self.device)
        self.text_only = False

    def forward_queries(self, queries: List[str], **kwargs) -> torch.Tensor:
        output = self.model.encode_text(queries)
        return torch.tensor(output).to(self.device)

    def forward_documents(self, documents: List[Image.Image], **kwargs) -> torch.Tensor:
        output = self.model.encode_image(documents)
        return torch.tensor(output).to(self.device)

    def get_scores(
        self, queries: List[str], documents: List[Image.Image | str], batch_query: int, batch_doc: int
    ) -> torch.Tensor:

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
