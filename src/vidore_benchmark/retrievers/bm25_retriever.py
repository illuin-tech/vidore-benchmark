from typing import Dict, List, Optional, Union, cast

import numpy as np
import torch
from PIL import Image

from vidore_benchmark.retrievers.registry_utils import register_vision_retriever
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever
from vidore_benchmark.utils.torch_utils import get_torch_device


@register_vision_retriever("bm25")
class BM25Retriever(VisionRetriever):
    def __init__(self, device: str = "auto"):
        super().__init__()

        try:
            from nltk.corpus import stopwords
            from nltk.tokenize import word_tokenize
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                'Install the missing dependencies with `pip install "vidore-benchmark[bm25]"` to use BM25Retriever.'
            )

        self.stopwords = stopwords
        self.word_tokenize = word_tokenize
        self.bm25_okapi_class = BM25Okapi

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

        # Sanity check: `passages` must be a list of filepaths (strings)
        if passages and not all(isinstance(doc, str) for doc in passages):
            raise ValueError("`passages` must be a list of filepaths (strings)")
        passages = cast(List[str], passages)

        queries_dict = {idx: query for idx, query in enumerate(queries)}
        tokenized_queries = self.preprocess_text(queries_dict)

        corpus = {idx: passage for idx, passage in enumerate(passages)}
        tokenized_corpus = self.preprocess_text(corpus)
        output = self.bm25_okapi_class(tokenized_corpus)

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
        stop_words = set(self.stopwords.words("english"))
        tokenized_list = [
            [word.lower() for word in self.word_tokenize(sentence) if word.isalnum() and word.lower() not in stop_words]
            for sentence in passages.values()
        ]
        return tokenized_list
