from vidore_benchmark.retrievers.vision_retriever import VisionRetriever
import torch
from typing import List
from PIL import Image
from vidore_benchmark.retrievers.utils.register_models import register_vision_retriever
from vidore_benchmark.utils.torch_utils import get_torch_device
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from datasets import Dataset
import numpy as np
from typing import Dict, List, Tuple, cast


@register_vision_retriever("bm25")
class BM25Retriever(VisionRetriever):
    def __init__(self, device: str = "auto"):
        super().__init__()
        self.device = get_torch_device(device)

    @property
    def use_visual_embedding(self) -> bool:
        return False

    def forward_queries(self, queries, **kwargs) -> torch.Tensor:
        # return dummy tensor - not used
        return torch.tensor([])

    def forward_documents(self, documents: List[str], **kwargs) -> torch.Tensor:
        # return dummy tensor - not used
        return torch.tensor([])

    def get_scores(
        self, queries: List[str], documents: List[str | Image.Image], batch_query: int, batch_doc: int, **kwargs
    ) -> torch.Tensor:

        queries_dict = {idx: query for idx, query in enumerate(queries)}
        tokenized_queries = self.preprocess_text(queries_dict)

        corpus = {idx: passage for idx, passage in enumerate(documents)}
        tokenized_corpus = self.preprocess_text(corpus)
        output = BM25Okapi(tokenized_corpus)

        scores = []
        for query in tokenized_queries:
            score = output.get_scores(query)
            scores.append(score)

        scores = torch.tensor(np.array(scores))

        assert scores.shape == (len(queries), len(documents))

        return scores

    def preprocess_text(self, documents: dict):
        """
        Basic preprocessing of the text data in english : remove stopwords, punctuation, lowercase all the words

        return the tokenized list of words
        """
        stop_words = set(stopwords.words("english"))
        tokenized_list = [
            [
                word.lower()
                for word in word_tokenize(sentence, language="french")
                if word.isalnum() and word.lower() not in stop_words
            ]
            for sentence in documents.values()
        ]

        return tokenized_list
