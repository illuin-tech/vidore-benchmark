from vidore_benchmark.retrievers.vision_retriever import VisionRetriever
import torch
from typing import List
from PIL import Image
from vidore_benchmark.retrievers.utils.register_models import register_text_retriever
from vidore_benchmark.utils.iter_utils import batched
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel

@register_text_retriever("BAAI/bge-m3")
class BGEM3(VisionRetriever):
    def __init__(self, *args, **kwargs):
        self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        self.text_only = True
        self.is_multi_vector = False

    def forward_queries(self, queries, **kwargs) -> torch.Tensor:
        output = self.model.encode(queries, max_length=512)["dense_vecs"]
        return torch.tensor(output)

    def forward_documents(self, documents: List[str], **kwargs) -> torch.Tensor:
        output = self.model.encode(documents)["dense_vecs"]
        return torch.tensor(output)

    def get_scores(self, 
                    queries : List[str],
                    documents : List[str | Image.Image], 
                    batch_query : int, 
                    batch_doc : int) -> torch.Tensor:

        list_emb_queries: List[torch.Tensor] = []
        for query_batch in tqdm(batched(queries, batch_query), desc="Query batch", total=len(queries)//batch_query):
            query_embeddings = self.forward_queries(query_batch)
            list_emb_queries.append(query_embeddings)

        list_emb_documents: List[torch.Tensor] = []
        for doc_batch in tqdm(batched(documents, batch_doc), desc="Document batch", total=len(documents)//batch_doc):
            doc_embeddings = self.forward_documents(doc_batch) # type: ignore
            list_emb_documents.append(doc_embeddings)
        
        emb_queries = torch.cat(list_emb_queries, dim=0)
        emb_documents = torch.cat(list_emb_documents, dim=0)

        scores = torch.einsum("bd,cd->bc", emb_queries, emb_documents)

        assert scores.shape == (emb_queries.shape[0], emb_documents.shape[0])

        return scores