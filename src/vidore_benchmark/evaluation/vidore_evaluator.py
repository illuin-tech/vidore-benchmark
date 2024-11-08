from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import torch
from colpali_engine.trainer.eval_utils import CustomRetrievalEvaluator
from datasets import Dataset
from tqdm import tqdm
from transformers import set_seed

from vidore_benchmark.compression.token_pooling import BaseEmbeddingPooler
from vidore_benchmark.retrievers.bm25_retriever import BM25Retriever
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever
from vidore_benchmark.utils.iter_utils import batched

set_seed(42)


class BEIRDataset(TypedDict):
    corpus: Dataset
    queries: Dataset
    qrels: Dataset


class ViDoReEvaluator:
    def __init__(
        self,
        vision_retriever: VisionRetriever,
        embedding_pooler: Optional[BaseEmbeddingPooler] = None,
    ):
        self.vision_retriever = vision_retriever
        self.embedding_pooler = embedding_pooler

        # Dataset column names
        self.query_column = "query"
        self.passage_column = "image" if self.vision_retriever.use_visual_embedding else "text_description"
        self.passage_filename_column = "image_filename"

    def evaluate_dataset(
        self,
        ds: Union[Dataset, BEIRDataset],
        ds_format: str,
        batch_query: int,
        batch_passage: int,
        batch_score: Optional[int] = None,
    ) -> Dict[str, Optional[float]]:
        if ds_format == "qa":
            if not isinstance(ds, Dataset):
                raise ValueError("QA dataset should be of type `Dataset`")
            results = self.evaluate_qa_dataset(
                ds=ds,
                batch_query=batch_query,
                batch_passage=batch_passage,
                batch_score=batch_score,
            )
        elif ds_format == "beir":
            raise NotImplementedError
        else:
            raise ValueError(f"Unsupported ds_format: {ds_format}")

        return results

    def evaluate_qa_dataset(
        self,
        ds: Dataset,
        batch_query: int,
        batch_passage: int,
        batch_score: Optional[int] = None,
    ) -> Dict[str, Optional[float]]:
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
            deduped_queries=deduped_queries,
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

    def evaluate_beir_dataset(
        self,
        ds: BEIRDataset,
        batch_query: int,
        batch_passage: int,
        batch_score: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        TODO: add documentation
        """
        # Load datasets
        ds_corpus = ds["corpus"]
        ds_queries = ds["queries"]
        ds_qrels = ds["qrels"]

        # Get image data
        image_ids: List[int] = list(ds_corpus["corpus-id"])
        # images: List[Image.Image] = list(ds_corpus["image"])

        # Get deduplicated query data
        query_ids: List[int] = ds_queries["query-id"]
        deduped_queries = self._get_deduped_queries(ds_queries["query"])
        if len(deduped_queries) == 0:
            raise ValueError("No valid queries found in the dataset. Check if the queries are all set to `None`.")

        # Get query relevance data
        qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
        for qrel in ds_qrels:
            # NOTE: The IDs are stored as integers in the dataset.
            query_id = str(qrel["query-id"])
            corpus_id = str(qrel["corpus-id"])
            qrels[query_id][corpus_id] = int(qrel["score"])

        # Edge case: using the BM25Retriever
        # FIXME: ...
        if isinstance(self.vision_retriever, BM25Retriever):
            passages = ds[passage_column_name]
            scores = self.vision_retriever.get_scores_bm25(queries=queries, passages=passages)
            qrels = self._get_qrels_from_qa_dataset(ds=ds)
            results = self._get_retrieval_results(
                ds=ds,
                deduped_queries=queries,
                scores=scores,
            )
            metrics = self.compute_retrieval_scores(qrels, results)
            return metrics

        # Get the embeddings for the queries and passages
        query_embeddings, passage_embeddings = self._get_query_and_passage_embeddings(
            ds=ds_corpus,
            deduped_queries=deduped_queries,
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

    def _get_query_and_passage_embeddings(
        self,
        ds: Dataset,
        deduped_queries: List[str],
        batch_query: int,
        batch_passage: int,
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], Union[torch.Tensor, List[torch.Tensor]]]:
        # Get the embeddings for the queries
        query_embeddings = self.vision_retriever.forward_queries(deduped_queries, batch_size=batch_query)

        # Get the embeddings for the passages
        passage_embeddings: List[torch.Tensor] = []

        # NOTE: To prevent overloading the RAM for large datasets, we will load the passages (images)
        # that will be fed to the model in batches (this should be fine for queries as their memory footprint
        # is negligible. This optimization is about efficient data loading, and is not related to the model's
        # forward pass which is also batched.

        dataloader_prebatch_size = 10 * batch_passage

        for passage_batch in tqdm(
            batched(ds, n=dataloader_prebatch_size),
            desc="Dataloader pre-batching",
            total=math.ceil(len(ds) / (dataloader_prebatch_size)),
        ):
            passages: List[Any] = [db[self.passage_column] for db in passage_batch]
            batch_emb_passages = self.vision_retriever.forward_passages(passages, batch_size=batch_passage)
            if isinstance(batch_emb_passages, torch.Tensor):
                batch_emb_passages = list(torch.unbind(batch_emb_passages))
                passage_embeddings.extend(batch_emb_passages)
            else:
                passage_embeddings.extend(batch_emb_passages)

        # Pool the document embeddings
        if self.embedding_pooler is not None:
            for idx, emb_document in tqdm(
                enumerate(passage_embeddings), total=len(passage_embeddings), desc="Pooling embeddings..."
            ):
                emb_document, _ = self.embedding_pooler.pool_embeddings(emb_document)
                passage_embeddings[idx] = emb_document

        return query_embeddings, passage_embeddings

    @staticmethod
    def compute_retrieval_scores(
        qrels: Dict[str, Dict[str, int]],
        results: Dict[str, Dict[str, float]],
        **kwargs,
    ) -> Dict[str, Optional[float]]:
        """
        Compute the MTEB metrics for retrieval.
        """

        mteb_evaluator = CustomRetrievalEvaluator()

        ndcg, _map, recall, precision, naucs = mteb_evaluator.evaluate(
            qrels=qrels,
            results=results,
            k_values=mteb_evaluator.k_values,
            ignore_identical_ids=kwargs.get("ignore_identical_ids", True),
        )

        mrr = mteb_evaluator.evaluate_custom(qrels, results, mteb_evaluator.k_values, "mrr")

        scores = {
            **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
            **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
            **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
            **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
            **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr[0].items()},
            **{f"naucs_at_{k.split('@')[1]}": v for (k, v) in naucs.items()},
        }

        return scores

    def _get_retrieval_results(
        self,
        ds: Dataset,
        deduped_queries: List[str],
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
