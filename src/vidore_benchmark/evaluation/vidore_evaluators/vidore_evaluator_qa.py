from __future__ import annotations

from typing import Dict, List, Optional

import torch
from datasets import Dataset

from vidore_benchmark.compression.token_pooling import BaseEmbeddingPooler
from vidore_benchmark.evaluation.vidore_evaluators.base_vidore_evaluator import BaseViDoReEvaluator
from vidore_benchmark.retrievers.base_vision_retriever import BaseVisionRetriever
from vidore_benchmark.retrievers.bm25_retriever import BM25Retriever


class ViDoReEvaluatorQA(BaseViDoReEvaluator):
    """
    Evaluator for the ViDoRe benchmark for datasets with a question-answering (QA) format, i.e. where each
    row in the dataset contains an optional query and a passage (image or text).
    """

    def __init__(
        self,
        vision_retriever: BaseVisionRetriever,
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
    ) -> Dict[str, Optional[float]]:
        # Get the deduplicated queries
        deduped_queries = self._deduplicate_queries(ds[self.query_column])
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
        query_embeddings = self._get_query_embeddings(
            ds=ds_deduped_queries,
            query_column=self.query_column,
            batch_query=batch_query,
        )
        passage_embeddings = self._get_passage_embeddings(
            ds=ds_passages,
            passage_column=self.passage_column,
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

    def _deduplicate_queries(self, queries: List[str]) -> List[str]:
        """
        Remove duplicate queries and `None` queries (i.e. passages with no associated query).

        Important notes:
        - This logic differs from the eval in `colpali-engine` where duplicates are NOT removed.
        - For fairness wrt externally evaluated retrievers since bug, we maintain this behavior and remove duplicates.
          This slightly boosts scores on some datasets, e.g. DocVQA typically.

        Args:
            queries (List[str]): The list of queries to deduplicate.

        Returns:
            (List[str]): The deduplicated queries.
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
        Get the retrieval results from the model's scores, i.e. the retrieval scores
        for each document for each query.

        Args:
            ds(Dataset): The dataset containing the queries and passages.
            deduped_queries(List[str]): The deduplicated queries.
            scores(torch.Tensor): The similarity scores between queries and passages (shape: n_queries, n_passages).

        Returns:
            (Dict[str, Dict[str, float]]): The retrieval results.

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
        Get the query relevance judgments (qrels) from a dataset (QA format).

        Args:
            ds (Dataset): The dataset (QA format) containing the queries and passages.

        Returns:
            Dict[str, Dict[str, int]]: The query relevance judgments (qrels).

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

        deduped_queries = self._deduplicate_queries(ds[self.query_column])
        for query in deduped_queries:
            qrels[query] = {query_to_filename[query]: 1}

        return qrels
