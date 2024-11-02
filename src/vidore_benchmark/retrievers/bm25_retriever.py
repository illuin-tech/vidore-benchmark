from typing import Dict, List, Optional, Union, cast

import numpy as np
import torch
from colpali_engine.utils.torch_utils import get_torch_device
from PIL import Image

from vidore_benchmark.retrievers.registry_utils import register_vision_retriever
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever


@register_vision_retriever("bm25")
class BM25Retriever(VisionRetriever):
    def __init__(self, device: str = "auto"):
        super().__init__()
        self.device = get_torch_device(device)

    @property
    def use_visual_embedding(self) -> bool:
        return False

    def forward_queries(self, queries, batch_size: int, **kwargs) -> List[torch.Tensor]:
        raise NotImplementedError("BM25Retriever only need get_scores_bm25 method.")

    def forward_documents(self, documents: List[str], batch_size: int, **kwargs) -> List[torch.Tensor]:
        raise NotImplementedError("BM25Retriever only need get_scores_bm25 method.")

    def get_scores(
        self,
        query_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        passage_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError("Please use the `get_scores_bm25` method instead.")

    def get_scores_bm25(
        self,
        queries: List[str],
        documents: Union[List[Image.Image], List[str]],
        **kwargs,
    ) -> torch.Tensor:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("Please install the `rank-bm25` package to use BM25Retriever.")

        # Sanity check: `documents` must be a list of filepaths (strings)
        if documents and not all(isinstance(doc, str) for doc in documents):
            raise ValueError("Documents must be a list of filepaths (strings)")
        documents = cast(List[str], documents)

        queries_dict = {idx: query for idx, query in enumerate(queries)}
        tokenized_queries = self.preprocess_text(queries_dict)

        corpus = {idx: passage for idx, passage in enumerate(documents)}
        tokenized_corpus = self.preprocess_text(corpus)
        output = BM25Okapi(tokenized_corpus)

        scores = []
        for query in tokenized_queries:
            score = output.get_scores(query)
            scores.append(score)

        scores = torch.tensor(np.array(scores))  # (num_queries, num_docs)

        return scores

    def preprocess_text(self, documents: Dict[int, str]) -> List[List[str]]:
        """
        Basic preprocessing of the text data:
        - remove stopwords
        - punctuation
        - lowercase all the words.
        """
        try:
            from nltk.corpus import stopwords
            from nltk.tokenize import word_tokenize
        except ImportError:
            raise ImportError("Please install the `nltk` package to use BM25Retriever.")

        stop_words = set(stopwords.words("english"))
        tokenized_list = [
            [word.lower() for word in word_tokenize(sentence) if word.isalnum() and word.lower() not in stop_words]
            for sentence in documents.values()
        ]
        return tokenized_list
