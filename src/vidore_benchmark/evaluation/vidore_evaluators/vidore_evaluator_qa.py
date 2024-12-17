from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional

import torch
from datasets import Dataset
from tqdm import tqdm

from vidore_benchmark.compression.token_pooling import BaseEmbeddingPooler
from vidore_benchmark.evaluation.vidore_evaluators.base_vidore_evaluator import BaseViDoReEvaluator
from vidore_benchmark.retrievers.base_vision_retriever import BaseVisionRetriever
from vidore_benchmark.retrievers.bm25_retriever import BM25Retriever
from vidore_benchmark.utils.image_utils import hash_image


class ViDoReEvaluatorQA(BaseViDoReEvaluator):
    """
    Evaluator for the ViDoRe benchmark for datasets with a question-answering (QA) format, i.e. where each
    row in the dataset contains an optional query and a passage (image or text).

    **IMPORTANT**: The old ViDoRe evaluation (<5.0.0) had a bug when computing `qrels`. This would slightly boost scores
    on some datasets (e.g. DocVQA). To reproduce the old behavior, set `is_legacy=True`.
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
        dataloader_prebatch_size: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Optional[float]]:
        ds_passages = ds.remove_columns(
            [col for col in ds.column_names if col not in [self.passage_column, self.passage_filename_column]]
        )

        # Add image hashing and deduplication
        if self.vision_retriever.use_visual_embedding:
            ds_passages = ds_passages.map(
                lambda x: {"image_hash": hash_image(x[self.passage_column])},
                desc="Hashing images for deduplication...",
            )
            ds_passages = self._deduplicate_dataset_rows(ds=ds_passages, target_column="image_hash")
            ds_passages = ds_passages.remove_columns(["image_hash"])

        ds_queries = ds.remove_columns([col for col in ds.column_names if col != self.query_column])
        ds_queries = self._deduplicate_dataset_rows(ds=ds_queries, target_column=self.query_column)

        if len(ds_queries) == 0:
            raise ValueError("No valid queries found in the dataset. Check if the queries are all set to `None`.")

        # Edge case: using the BM25Retriever
        if isinstance(self.vision_retriever, BM25Retriever):
            scores = self.vision_retriever.get_scores_bm25(
                queries=ds_queries[self.query_column],
                passages=ds_passages[self.passage_column],
            )

            qrels = self._get_qrels_from_qa_dataset(ds=ds)
            results = self._get_retrieval_results(
                ds_passages=ds_passages,
                ds_deduped_queries=ds_queries,
                scores=scores,
            )

            metrics = self.compute_retrieval_scores(qrels=qrels, results=results)

            return metrics

        # Get the embeddings for the queries and passages
        query_embeddings = self._get_query_embeddings(
            ds=ds_queries,
            query_column=self.query_column,
            batch_query=batch_query,
            dataloader_prebatch_size=dataloader_prebatch_size,
        )
        passage_embeddings = self._get_passage_embeddings(
            ds=ds_passages,
            passage_column=self.passage_column,
            batch_passage=batch_passage,
            dataloader_prebatch_size=dataloader_prebatch_size,
        )

        # Use token pooling (optional)
        if self.embedding_pooler is not None:
            for idx, passage_embedding in tqdm(
                enumerate(passage_embeddings), total=len(passage_embeddings), desc="Pooling embeddings..."
            ):
                passage_embedding, _ = self.embedding_pooler.pool_embeddings(passage_embedding)
                passage_embeddings[idx] = passage_embedding

        # Get the similarity scores
        scores = self.vision_retriever.get_scores(
            query_embeddings=query_embeddings,
            passage_embeddings=passage_embeddings,
            batch_size=batch_score,
        )

        # Get the relevant passages and results
        qrels = self._get_qrels_from_qa_dataset(ds=ds)
        results = self._get_retrieval_results(
            ds_passages=ds_passages,
            ds_deduped_queries=ds_queries,
            scores=scores,
        )

        # Compute the MTEB metrics
        metrics = self.compute_retrieval_scores(qrels=qrels, results=results)

        return metrics

    def _deduplicate_dataset_rows(self, ds: Dataset, target_column: str) -> Dataset:
        """
        Remove duplicate rows from a dataset based on values in a target column.

        Args:
            ds (Dataset): The dataset to deduplicate.
            target_column (str): The column to use for deduplication.

        Returns:
            Dataset: The deduplicated dataset.
        """
        if target_column not in ds.column_names:
            raise ValueError(f"Column '{target_column}' not found in dataset.")

        seen_values = set()
        keep_mask = []

        for value in ds[target_column]:
            if value is None:
                keep_mask.append(False)
                continue

            if value not in seen_values:
                seen_values.add(value)
                keep_mask.append(True)
            else:
                keep_mask.append(False)

        return ds.select([i for i, keep in enumerate(keep_mask) if keep])

    def _get_retrieval_results(
        self,
        ds_passages: Dataset,
        ds_deduped_queries: Dataset,
        scores: torch.Tensor,
    ) -> Dict[str, Dict[str, float]]:
        """
        Get the retrieval results from the model's scores, i.e. the retrieval scores
        for each document for each query.

        Args:
            ds_passages (Dataset): The dataset containing the passages.
            ds_queries (Dataset): The dataset containing the deduplicated queries.
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
            passage_id: image_filename
            for passage_id, image_filename in enumerate(ds_passages[self.passage_filename_column])
        }

        # Placeholders
        results: Dict[str, Dict[str, float]] = {}

        for query, score_per_query in zip(ds_deduped_queries[self.query_column], scores):
            for doc_idx, score in enumerate(score_per_query):
                filename = passage_id_to_filename[doc_idx]
                score_passage = float(score.item())

                if query in results:
                    current_score = results[query].get(filename, 0)
                    results[query][filename] = max(current_score, score_passage)
                else:
                    results[query] = {filename: score_passage}

        return results

    def _get_qrels_from_qa_dataset(
        self,
        ds: Dataset,
    ) -> Dict[str, Dict[str, int]]:
        """
        Get the query relevance judgments (qrels) from a dataset (QA format).

        Args:
            ds (Dataset): The dataset containing the queries and passages.

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
        if self.passage_filename_column not in ds.column_names:
            raise ValueError(f"Passage filename column name '{self.passage_filename_column}' not found in the dataset.")
        if self.query_column not in ds.column_names:
            raise ValueError(f"Query column name '{self.query_column}' not found in the dataset.")

        qrels: Dict[str, Dict[str, int]] = defaultdict(dict)

        for query, passage_filename in zip(ds[self.query_column], ds[self.passage_filename_column]):
            if query is not None and query in ds[self.query_column]:
                qrels[query][passage_filename] = 1

        return qrels
