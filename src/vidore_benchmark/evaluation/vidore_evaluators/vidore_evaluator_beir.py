from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, TypedDict

import torch
from datasets import Dataset

from vidore_benchmark.evaluation.vidore_evaluators.base_vidore_evaluator import BaseViDoReEvaluator
from vidore_benchmark.retrievers.base_vision_retriever import BaseVisionRetriever
from vidore_benchmark.retrievers.bm25_retriever import BM25Retriever


class BEIRDataset(TypedDict):
    """
    BEIR dataset type. A BEIR dataset must contain 3 subsets:
        corpus: The dataset containing the corpus of documents.
        queries: The dataset containing the queries.
        qrels: The dataset containing the query relevance scores.

    Notes:
    - `qrels` follows the TREC format, where the structure is `{query_id: {doc_id: relevance_score}}`.
    - `relevance_score` is an integer indicating the relevance of the document to the query. For each query i,
    the relevance scores are integers in the range [0, N_i], where the higher the score, the more relevant
    the document is to the given query.
    """

    corpus: Dataset
    queries: Dataset
    qrels: Dataset


class ViDoReEvaluatorBEIR(BaseViDoReEvaluator):
    """
    Evaluator for the ViDoRe benchmark for datasets with a BEIR format, i.e. where each
    dataset contains 3 subsets:
        corpus: The dataset containing the corpus of documents.
        queries: The dataset containing the queries.
        qrels: The dataset containing the query relevance scores.

    **Important**: Do NOT use this evaluator for the ViDoRe (v1) leaderboard as the handling of duplicates
    slightly differs from the `ViDoReEvaluatorQA` evaluator.
    """

    def __init__(
        self,
        vision_retriever: BaseVisionRetriever,
        corpus_id_column: Optional[str] = None,
        query_id_column: Optional[str] = None,
        query_column: Optional[str] = None,
        passage_column: Optional[str] = None,
        score_column: Optional[str] = None,
    ):
        super().__init__(vision_retriever=vision_retriever)

        # Dataset column names
        self.corpus_id_column = corpus_id_column if corpus_id_column else "corpus-id"
        self.query_id_column = query_id_column if query_id_column else "query-id"
        self.query_column = query_column if query_column else "query"
        if passage_column:
            self.passage_column = passage_column
        else:
            self.passage_column = "image" if self.vision_retriever.use_visual_embedding else "text_description"
        self.score_column = score_column if score_column else "score"

    def evaluate_dataset(
        self,
        ds: BEIRDataset,
        batch_query: int,
        batch_passage: int,
        batch_score: Optional[int] = None,
        dataloader_prebatch_query: Optional[int] = None,
        dataloader_prebatch_passage: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Optional[float]]:
        """
        Evaluate the given BEIR dataset.

        Args:
            ds (BEIRDataset): The dataset to evaluate.
            batch_query (int): The batch size for processing queries.
            batch_passage (int): The batch size for processing passages.
            batch_score (Optional[int]): The batch size for computing similarity scores.
            dataloader_prebatch_query (Optional[int]): The number of queries to pre-batch before processing.
            dataloader_prebatch_passage (Optional[int]): The number of passages to pre-batch before processing.
        """
        # Load datasets
        ds_corpus = ds["corpus"]
        ds_queries = ds["queries"]
        ds_qrels = ds["qrels"]

        # Get image data
        passage_ids: List[str] = [str(elt) for elt in ds_corpus[self.corpus_id_column]]

        # Get query data
        query_ids: List[str] = [str(elt) for elt in ds_queries[self.query_id_column]]

        # Get query relevance data
        qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
        for qrel in ds_qrels:
            # Cast to str to handle int query IDs:
            query_id = str(qrel[self.query_id_column])
            corpus_id = str(qrel[self.corpus_id_column])

            qrels[query_id][corpus_id] = qrel[self.score_column]

        # Edge case: using the BM25Retriever
        if isinstance(self.vision_retriever, BM25Retriever):
            passages = ds_corpus[self.passage_column]
            queries: List[str] = ds_queries[self.query_column]

            scores = self.vision_retriever.get_scores_bm25(
                queries=queries,
                passages=passages,
            )
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
            dataloader_prebatch_size=dataloader_prebatch_query,
        )
        passage_embeddings = self._get_passage_embeddings(
            ds=ds_corpus,
            passage_column=self.passage_column,
            batch_passage=batch_passage,
            dataloader_prebatch_size=dataloader_prebatch_passage,
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
            passage_ids=passage_ids,
            scores=scores,
        )

        # Compute the MTEB metrics
        metrics = self.compute_retrieval_scores(
            qrels=qrels,
            results=results,
            ignore_identical_ids=False,
        )

        return metrics

    def _get_retrieval_results(
        self,
        query_ids: List[str],
        passage_ids: List[str],
        scores: torch.Tensor,
    ) -> Dict[str, Dict[str, float]]:
        """
        Get the retrieval results from the model's scores, i.e. the retrieval scores for each passage for each query.

        Args:
            query_ids (List[str]): The list of query IDs.
            passage_ids (List[str]): The list of passage IDs.
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

                if query_id in results:
                    current_score = results[query_id].get(image_id, 0)
                    results[query_id][image_id] = max(current_score, score_passage)
                else:
                    results[query_id] = {image_id: score_passage}

        return results
