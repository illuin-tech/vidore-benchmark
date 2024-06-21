from vidore_benchmark.retrievers.vision_retriever import VisionRetriever
from vidore_benchmark.retrievers.utils.register_models import register_text_retriever
import torch
from vidore_benchmark.utils.iter_utils import batched
from typing import List
from tqdm import tqdm
from PIL import Image


@register_text_retriever("dummy")
class DummyRetriever(VisionRetriever):
    def __init__(self, *args, **kwargs):
        self.text_only = False

    def forward_queries(self, queries, **kwargs):
        return torch.randn(len(queries), 512)

    def forward_documents(self, documents, **kwargs):
        return torch.randn(len(documents), 512)

    def get_scores(self, queries: List[str], documents: List[str | Image.Image], batch_query, batch_doc):

        scores = torch.randn(len(queries), len(documents))
        return scores
