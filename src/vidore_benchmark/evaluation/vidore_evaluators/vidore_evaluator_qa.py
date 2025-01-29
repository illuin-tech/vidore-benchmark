from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import torch
from datasets import Dataset
from tqdm import tqdm

from vidore_benchmark.compression.token_pooling import BaseEmbeddingPooler
from vidore_benchmark.evaluation.vidore_evaluators.base_vidore_evaluator import BaseViDoReEvaluator
from vidore_benchmark.retrievers.base_vision_retriever import BaseVisionRetriever
from vidore_benchmark.retrievers.bm25_retriever import BM25Retriever
from vidore_benchmark.utils.data_utils import deduplicate_dataset_rows


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
        self.id_column = "id"
        self.image_hash_column = "image_hash"

    def evaluate_dataset(
        self,
        ds: Dataset,
        batch_query: int,
        batch_passage: int,
        batch_score: Optional[int] = None,
        dataloader_prebatch_size: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Optional[float]]:
        # Preprocess the dataset, get qrels, and deduplicate the queries and passages
        ds = ds.map(lambda example, idx: {self.id_column: idx}, with_indices=True)

        ds_passages = ds.remove_columns(
            [col for col in ds.column_names if col not in [self.passage_column, self.image_hash_column, self.id_column]]
        )
        ds_queries = ds.remove_columns(
            [col for col in ds.column_names if col not in [self.query_column, self.id_column]]
        )
        ds_queries = deduplicate_dataset_rows(ds=ds_queries, target_column=self.query_column)

        passage_ids: List[int] = list(ds_passages[self.id_column])
        query_ids: List[int] = ds_queries[self.id_column]

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
                query_ids=query_ids,
                passage_ids=passage_ids,
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

        # Get the relevant query relevances (qrels) and results
        qrels = self._get_qrels_from_qa_dataset(ds=ds)
        results = self._get_retrieval_results(
            query_ids=query_ids,
            passage_ids=passage_ids,
            scores=scores,
        )

        # Compute the MTEB metrics
        metrics = self.compute_retrieval_scores(qrels=qrels, results=results)

        return metrics

    def _get_retrieval_results(
        self,
        query_ids: List[int],
        passage_ids: List[int],
        scores: torch.Tensor,
    ) -> Dict[str, Dict[str, float]]:
        """
        Get the retrieval results from the model's scores, i.e. the retrieval scores for each passage for each query.

        Args:
            query_ids (List[int]): The list of query IDs.
            passage_ids (List[int]): The list of passage IDs.
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
        results: Dict[str, Dict[str, float]] = {}

        for query_idx, query_id in enumerate(query_ids):
            for image_idx, score in enumerate(scores[query_idx]):
                image_id = passage_ids[image_idx]
                score_passage = float(score.item())

                if str(query_id) in results:
                    current_score = results[str(query_id)].get(str(image_id), 0)
                    results[str(query_id)][str(image_id)] = max(current_score, score_passage)
                else:
                    results[str(query_id)] = {str(image_id): score_passage}

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
                "0": {"1": 1},  # query_id: {passage_id: relevance_score}
                "1": {"2": 1},
                ...
            }
            ```
        """
        if self.id_column not in ds.column_names:
            raise ValueError(f"ID column name '{self.id_column}' not found in the dataset.")
        if self.query_column not in ds.column_names:
            raise ValueError(f"Query column name '{self.query_column}' not found in the dataset.")

        qrels: Dict[str, Dict[str, int]] = defaultdict(dict)

        # Legacy behavior (bug): only keep the last occurrence of a query.
        for query, passage_filename in zip(ds[self.query_column], ds[self.passage_filename_column]):
            if query is not None and query in ds[self.query_column]:
                qrels[query] = {passage_filename: 1}

        return qrels
