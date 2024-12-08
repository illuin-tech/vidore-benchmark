from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch
from datasets import Dataset
from tqdm import tqdm

from vidore_benchmark.compression.token_pooling import BaseEmbeddingPooler
from vidore_benchmark.evaluation.eval_utils import CustomRetrievalEvaluator
from vidore_benchmark.retrievers.base_vision_retriever import BaseVisionRetriever
from vidore_benchmark.utils.iter_utils import batched

logger = logging.getLogger(__name__)


class BaseViDoReEvaluator(ABC):
    """
    Base evaluator for the ViDoRe benchmark.

    Args:
        vision_retriever (BaseVisionRetriever): The vision retriever used to retrieve the embeddings.
        embedding_pooler (Optional[BaseEmbeddingPooler]): The embedding pooler used to pool the passage embeddings.
    """

    def __init__(
        self,
        vision_retriever: BaseVisionRetriever,
        embedding_pooler: Optional[BaseEmbeddingPooler] = None,
    ):
        self.vision_retriever = vision_retriever
        self.embedding_pooler = embedding_pooler

    @abstractmethod
    def evaluate_dataset(
        self,
        ds: Any,
        batch_query: int,
        batch_passage: int,
        batch_score: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Optional[float]]:
        """
        Evaluate the model on a given dataset using the MTEB retrieval metrics.

        Args:
            ds (Dataset): The dataset to evaluate the model on.
            batch_query (int): The batch size for the queries.
            batch_passage (int): The batch size for the passages.
            batch_score (Optional[int]): The batch size for the scoring.
            **kwargs: Additional keyword arguments.

        Returns:
            (Dict[str, float]): The MTEB retrieval metrics.

        NOTE: The `ds` dataset should contain the following columns:
            query (str): The query text.
            image_filename (str): The filename of the image.
            (if `use_visual_embedding`) image (PIL.Image): The image of the document page.
            (if not `use_visual_embedding`) text_description (str): The text of the document page,
                plus eventual description of visual elements.
        """
        pass

    def _get_passage_embeddings(
        self,
        ds: Dataset,
        passage_column: str,
        batch_passage: int,
        dataloader_prebatch_size: Optional[int] = None,
        **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Get the passage embeddings for the given dataset using the retriever used during class
        instance creation.

        Args:
            ds (Dataset): The dataset containing the queries and passages.
            passage_column (str): The name of the column containing the passages (i.e. images or text chunks).
            batch_passage (int): The batch size for the passages.
            dataloader_prebatch_size (Optional[int]): The pre-batch size for the dataloader. If set, must be
                greater than or equal to `batch_passage`.

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: The passage embeddings.
        """
        passage_embeddings: List[torch.Tensor] = []

        # NOTE: To prevent overloading the RAM for large datasets, we will load the passages (images)
        # that will be fed to the model in batches (this should be fine for queries as their memory footprint
        # is negligible. This optimization is about efficient data loading, and is not related to the model's
        # forward pass which is also batched.

        if dataloader_prebatch_size is None:
            dataloader_prebatch_size = batch_passage
        if dataloader_prebatch_size < batch_passage:
            logger.warning(
                f"`dataloader_prebatch_size` ({dataloader_prebatch_size}) is smaller than `batch_passage` "
                f"({batch_passage}). Setting the pre-batch size to the passager batch size."
            )
            dataloader_prebatch_size = batch_passage

        for ds_batch in tqdm(
            batched(ds, n=dataloader_prebatch_size),
            desc="Dataloader pre-batching for passages",
            total=math.ceil(len(ds) / (dataloader_prebatch_size)),
        ):
            passages: List[Any] = [batch[passage_column] for batch in ds_batch]

            batch_embedding_passages = self.vision_retriever.forward_passages(
                passages=passages,
                batch_size=batch_passage,
            )

            if isinstance(batch_embedding_passages, torch.Tensor):
                batch_embedding_passages = list(torch.unbind(batch_embedding_passages.to("cpu")))
                passage_embeddings.extend(batch_embedding_passages)
            else:
                for embedding_passage in batch_embedding_passages:
                    passage_embeddings.append(embedding_passage.to("cpu"))

        return passage_embeddings

    @torch.no_grad()
    def _get_query_embeddings(
        self,
        ds: Dataset,
        query_column: str,
        batch_query: int,
        dataloader_prebatch_size: Optional[int] = None,
        **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Get the query embeddings for the input queries using the retriever used during class instance creation.

        Args:
            ds (Dataset): The dataset containing the queries and passages.
            query_column (str): The name of the column containing the queries.
            batch_query (int): The batch size for the queries.
            dataloader_prebatch_size (Optional[int]): The pre-batch size for the dataloader. If set, must be
                greater than or equal to `batch_query`.

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: The query embeddings.
        """
        query_embeddings: List[torch.Tensor] = []

        # NOTE: To prevent overloading the RAM for large datasets, we will load the passages (images)
        # that will be fed to the model in batches (this should be fine for queries as their memory footprint
        # is negligible. This optimization is about efficient data loading, and is not related to the model's
        # forward pass which is also batched.

        if dataloader_prebatch_size is None:
            dataloader_prebatch_size = batch_query
        if dataloader_prebatch_size < batch_query:
            logger.warning(
                f"`dataloader_prebatch_size` ({dataloader_prebatch_size}) is smaller than `batch_query` "
                f"({batch_query}). Setting the pre-batch size to the passager batch size."
            )
            dataloader_prebatch_size = batch_query

        for ds_batch in tqdm(
            batched(ds, n=dataloader_prebatch_size),
            desc="Dataloader pre-batching for queries",
            total=math.ceil(len(ds) / (dataloader_prebatch_size)),
        ):
            queries: List[Any] = [batch[query_column] for batch in ds_batch]
            batch_embedding_queries = self.vision_retriever.forward_queries(
                queries=queries,
                batch_size=batch_query,
            )

            if isinstance(batch_embedding_queries, torch.Tensor):
                batch_embedding_queries = list(torch.unbind(batch_embedding_queries.to("cpu")))
                query_embeddings.extend(batch_embedding_queries)
            else:
                for embedding_query in batch_embedding_queries:
                    query_embeddings.append(embedding_query.to("cpu"))

        return query_embeddings

    @staticmethod
    def compute_retrieval_scores(
        qrels: Dict[str, Dict[str, int]],
        results: Dict[str, Dict[str, float]],
        **kwargs,
    ) -> Dict[str, Optional[float]]:
        """
        Compute the MTEB retrieval metrics (NDCG, MAP, Recall, Precision, NDCG, MRR, NDCG, and NDCG).

        Args:
            qrels: A dictionary containing the degree of relevance between queries and documents,
                following the BEIR convention (0: irrelevant, 1: relevant).
            results: A dictionary containing the retrieval results, i.e. the retrieval
                scores for each document for each query.
                Example input:
                ```python
                {
                    "query_0": {"doc_i": 19.125, "doc_1": 18.75, ...},
                    "query_1": {"doc_j": 17.25, "doc_1": 16.75, ...},
                    ...
                }
                ```
            **kwargs: Additional keyword arguments.
        """
        mteb_evaluator = CustomRetrievalEvaluator()

        ndcg, _map, recall, precision, naucs = mteb_evaluator.evaluate(
            qrels=qrels,
            results=results,
            k_values=mteb_evaluator.k_values,
            ignore_identical_ids=kwargs.get("ignore_identical_ids", True),
        )

        mrr = mteb_evaluator.evaluate_custom(qrels, results, mteb_evaluator.k_values, "mrr")

        scores: Dict[str, Optional[float]] = {
            **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
            **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
            **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
            **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
            **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr[0].items()},
            **{f"naucs_at_{k.split('@')[1]}": v for (k, v) in naucs.items()},
        }

        return scores
