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

    def forward_passages(self, passages: List[str], batch_size: int, **kwargs) -> List[torch.Tensor]:
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
        passages: Union[List[Image.Image], List[str]],
        **kwargs,
    ) -> torch.Tensor:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("Please install the `rank-bm25` package to use BM25Retriever.")

        # Sanity check: `passages` must be a list of filepaths (strings)
        if passages and not all(isinstance(doc, str) for doc in passages):
            raise ValueError("`passages` must be a list of filepaths (strings)")
        passages = cast(List[str], passages)

        queries_dict = {idx: query for idx, query in enumerate(queries)}
        tokenized_queries = self.preprocess_text(queries_dict)

        corpus = {idx: passage for idx, passage in enumerate(passages)}
        tokenized_corpus = self.preprocess_text(corpus)
        output = BM25Okapi(tokenized_corpus)

        scores = []
        for query in tokenized_queries:
            score = output.get_scores(query)
            scores.append(score)

        scores = torch.tensor(np.array(scores))  # (n_queries, n_passages)

        return scores

    def preprocess_text(self, passages: Dict[int, str]) -> List[List[str]]:
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
            for sentence in passages.values()
        ]
        return tokenized_list
