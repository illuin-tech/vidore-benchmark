from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, TypedDict

import torch
from datasets import Dataset

from vidore_benchmark.compression.token_pooling import BaseEmbeddingPooler
from vidore_benchmark.evaluation.vidore_evaluator.vidore_evaluator_base import ViDoReEvaluatorBase
from vidore_benchmark.retrievers.bm25_retriever import BM25Retriever
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever


class BEIRDataset(TypedDict):
    corpus: Dataset
    queries: Dataset
    qrels: Dataset


class ViDoReEvaluatorBEIR(ViDoReEvaluatorBase):
    def __init__(
        self,
        vision_retriever: VisionRetriever,
        embedding_pooler: Optional[BaseEmbeddingPooler] = None,
    ):
        super().__init__(
            vision_retriever=vision_retriever,
            embedding_pooler=embedding_pooler,
        )

    def evaluate_dataset(
        self,
        ds: BEIRDataset,
        batch_query: int,
        batch_passage: int,
        batch_score: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Optional[float]]:
        """
        TODO: add documentation
        """
        # Load datasets
        ds_corpus = ds["corpus"]
        ds_queries = ds["queries"]
        ds_qrels = ds["qrels"]

        # Get image data
        image_ids: List[int] = list(ds_corpus["corpus-id"])

        # Get deduplicated query data
        query_ids: List[int] = ds_queries["query-id"]
        queries: List[str] = ds_queries["query"]

        # Get query relevance data
        qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
        for qrel in ds_qrels:
            # NOTE: The IDs are stored as integers in the dataset.
            query_id = str(qrel["query-id"])
            corpus_id = str(qrel["corpus-id"])
            qrels[query_id][corpus_id] = int(qrel["score"])

        # Edge case: using the BM25Retriever
        if isinstance(self.vision_retriever, BM25Retriever):
            passages = ds_corpus["text_description"]
            scores = self.vision_retriever.get_scores_bm25(
                queries=queries,
                passages=passages,
            )
            results = self._get_retrieval_results(
                query_ids=query_ids,
                image_ids=image_ids,
                scores=scores,
            )
            metrics = self.compute_retrieval_scores(qrels=qrels, results=results)
            return metrics

        # Get the embeddings for the queries and passages
        query_embeddings, passage_embeddings = self._get_query_and_passage_embeddings(
            ds=ds_corpus,
            passage_column="image",
            queries=queries,
            batch_query=batch_query,
            batch_passage=batch_passage,
        )

        # Get the similarity scores
        scores = self.vision_retriever.get_scores(
            query_embeddings=query_embeddings,
            passage_embeddings=passage_embeddings,
            batch_size=batch_score,
        )

        # Get the relevant passages and results
        results = self._get_retrieval_results(
            query_ids=query_ids,
            image_ids=image_ids,
            scores=scores,
        )

        # Compute the MTEB metrics
        metrics = self.compute_retrieval_scores(qrels=qrels, results=results)

        return metrics

    def _get_retrieval_results(
        self,
        query_ids: List[int],
        image_ids: List[int],
        scores: torch.Tensor,
    ) -> Dict[str, Dict[str, float]]:
        """
        Get the retrieval results from the model's scores.

        Outputs:
            results: Dict[str, Dict[str, float]]

        Example output:
            ```python
            {
                "query_0": {"doc_i": 19.125, "doc_1": 18.75, ...},
                "query_1": {"doc_j": 17.25, "doc_1": 16.75, ...},
                ...
            }
            ```
        """
        results: Dict[str, Dict[str, float]] = defaultdict(dict)

        for i, query_id in enumerate(query_ids):
            query_id = str(query_id)
            _, indices = torch.sort(scores[i], descending=True)
            for idx in indices:
                corpus_id = str(image_ids[idx])
                results[query_id][corpus_id] = scores[i][idx].item()

        return results
