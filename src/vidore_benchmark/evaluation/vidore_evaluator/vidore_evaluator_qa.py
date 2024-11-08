from __future__ import annotations

from typing import Dict, List, Optional

import torch
from datasets import Dataset
from transformers import set_seed

from vidore_benchmark.compression.token_pooling import BaseEmbeddingPooler
from vidore_benchmark.evaluation.vidore_evaluator.vidore_evaluator_base import ViDoReEvaluatorBase
from vidore_benchmark.retrievers.bm25_retriever import BM25Retriever
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever

set_seed(42)


class ViDoReEvaluatorQA(ViDoReEvaluatorBase):
    def __init__(
        self,
        vision_retriever: VisionRetriever,
        embedding_pooler: Optional[BaseEmbeddingPooler] = None,
    ):
        super().__init__(
            vision_retriever=vision_retriever,
            embedding_pooler=embedding_pooler,
        )

        # Dataset column names
        self.query_column = "query"
        self.passage_column = "image" if self.vision_retriever.use_visual_embedding else "text_description"
        self.passage_filename_column = "image_filename"

    def evaluate_dataset(
        self,
        ds: Dataset,
        batch_query: int,
        batch_passage: int,
        batch_score: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Evaluate the model on a given dataset using the MTEB metrics.

        NOTE: The dataset should contain the following columns:
        - query: the query text
        - image_filename: the filename of the image
        - image: the image (PIL.Image) if `use_visual_embedding` is True
        - text_description: the text description (i.e. the page caption or the text chunks) if
            `use_visual_embedding` is False
        """

        # Get the deduplicated queries
        deduped_queries = self._get_deduped_queries(ds[self.query_column])
        if len(deduped_queries) == 0:
            raise ValueError("No valid queries found in the dataset. Check if the queries are all set to `None`.")

        # Edge case: using the BM25Retriever
        if isinstance(self.vision_retriever, BM25Retriever):
            passages = ds["text_description"]
            scores = self.vision_retriever.get_scores_bm25(queries=deduped_queries, passages=passages)
            qrels = self._get_qrels_from_qa_dataset(ds=ds)
            results = self._get_retrieval_results(
                ds=ds,
                deduped_queries=deduped_queries,
                scores=scores,
            )
            metrics = self.compute_retrieval_scores(qrels, results)
            return metrics

        # Get the embeddings for the queries and passages
        query_embeddings, passage_embeddings = self._get_query_and_passage_embeddings(
            ds=ds,
            passage_column=self.passage_column,
            queries=deduped_queries,
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
        qrels = self._get_qrels_from_qa_dataset(ds=ds)
        results = self._get_retrieval_results(
            ds=ds,
            deduped_queries=deduped_queries,
            scores=scores,
        )

        # Compute the MTEB metrics
        metrics = self.compute_retrieval_scores(qrels=qrels, results=results)

        return metrics

    def _get_deduped_queries(self, queries: List[str]) -> List[str]:
        """
        Remove `None` queries (i.e. pages for which no question was generated) and duplicates.

        Notes:
        - This logic differs from the eval in `colpali-engine` where duplicates are NOT removed.
        - For fairness wrt externally evaluated retrievers since bug, we maintain this behavior and remove duplicates.
          This slightly boosts scores on some datasets, e.g. DocVQA typically.
        """
        seen_queries = set()
        deduped_queries: List[str] = []

        for query in queries:
            if query is not None and query not in seen_queries:
                deduped_queries.append(query)
                seen_queries.add(query)

        return deduped_queries

    def _get_retrieval_results(
        self,
        ds: Dataset,
        deduped_queries: List[str],
        scores: torch.Tensor,
    ) -> Dict[str, Dict[str, float]]:
        """
        Get the retrieval results from the model's scores, i.e. the retrieval
        scores for each document for each query.

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
        # Get the mapping
        passage_id_to_filename: Dict[int, str] = {
            passage_id: image_filename for passage_id, image_filename in enumerate(ds[self.passage_filename_column])
        }

        # Placeholders
        results: Dict[str, Dict[str, float]] = {}

        for query, score_per_query in zip(deduped_queries, scores):
            for doc_idx, score in enumerate(score_per_query):
                filename = passage_id_to_filename[doc_idx]
                score_passage = float(score.item())

                if query in results:
                    current_score = results[query].get(filename, 0)
                    results[query][filename] = max(current_score, score_passage)
                else:
                    results[query] = {filename: score_passage}

        return results

    def _get_qrels_from_qa_dataset(self, ds: Dataset) -> Dict[str, Dict[str, int]]:
        """
        Get the relevant passages (qrels) from a QA dataset.

        Returns:
            qrels: Dict[str, Dict[str, int]]

        Example output:
            ```python
            {
                "query_0": {"doc_0": 1},
                "query_1": {"doc_1": 1},
                ...
            }
            ```
        """
        # Sanity checks
        if self.query_column not in ds.column_names:
            raise ValueError(f"Query column name '{self.query_column}' not found in the dataset.")
        if self.passage_filename_column not in ds.column_names:
            raise ValueError(f"Passage filename column name '{self.passage_filename_column}' not found in the dataset.")

        # Placeholder
        qrels: Dict[str, Dict[str, int]] = {}

        # Get the mappings
        query_to_filename: Dict[str, str] = {
            query: image_filename
            for query, image_filename in zip(ds[self.query_column], ds[self.passage_filename_column])
        }

        deduped_queries = self._get_deduped_queries(ds[self.query_column])
        for query in deduped_queries:
            qrels[query] = {query_to_filename[query]: 1}

        return qrels
