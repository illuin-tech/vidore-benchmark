from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, TypedDict

import torch
from datasets import Dataset

from vidore_benchmark.compression.token_pooling import BaseEmbeddingPooler
from vidore_benchmark.evaluation.vidore_evaluators.base_vidore_evaluator import BaseViDoReEvaluator
from vidore_benchmark.retrievers.base_vision_retriever import BaseVisionRetriever
from vidore_benchmark.retrievers.bm25_retriever import BM25Retriever


class BEIRDataset(TypedDict):
    """
    BEIR dataset type. A BEIR dataset must contain 3 subsets:
        corpus: The dataset containing the corpus of documents.
        queries: The dataset containing the queries.
        qrels: The dataset containing the query relevance scores.

    Each subset is associated to a key with the same name.
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

    Paper reference for BEIR: https://doi.org/10.48550/arXiv.2104.08663
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
        self.corpus_id_column = "corpus-id"
        self.query_id_column = "query-id"
        self.query_column = "query"
        self.passage_column = "image" if self.vision_retriever.use_visual_embedding else "text_description"
        self.score_column = "score"

    def evaluate_dataset(
        self,
        ds: BEIRDataset,
        batch_query: int,
        batch_passage: int,
        batch_score: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Optional[float]]:
        # Load datasets
        ds_corpus = ds["corpus"]
        ds_queries = ds["queries"]
        ds_qrels = ds["qrels"]

        # Get image data
        image_ids: List[int] = list(ds_corpus[self.corpus_id_column])

        # Get deduplicated query data
        query_ids: List[int] = ds_queries[self.query_id_column]
        queries: List[str] = ds_queries[self.query_column]

        # Get query relevance data
        qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
        for qrel in ds_qrels:
            # NOTE: The IDs are stored as integers in the dataset.
            query_id = str(qrel[self.query_id_column])
            corpus_id = str(qrel[self.corpus_id_column])
            qrels[query_id][corpus_id] = int(qrel[self.score_column])

        # Edge case: using the BM25Retriever
        if isinstance(self.vision_retriever, BM25Retriever):
            passages = ds_corpus[self.passage_column]
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
            passage_column=self.passage_column,
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
        Get the retrieval results from the model's scores, i.e. the retrieval scores
        for each document for each query.

        Args:
            query_ids (List[int]): The list of query IDs.
            image_ids (List[int]): The list of image IDs.
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
                image_id = image_ids[image_idx]
                score_passage = float(score.item())

                if str(query_id) in results:
                    current_score = results[str(query_id)].get(str(image_id), 0)
                    results[str(query_id)][str(image_id)] = max(current_score, score_passage)
                else:
                    results[str(query_id)] = {str(image_id): score_passage}

        return results
