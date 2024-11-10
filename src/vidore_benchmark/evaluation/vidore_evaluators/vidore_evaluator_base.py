from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from colpali_engine.trainer.eval_utils import CustomRetrievalEvaluator
from datasets import Dataset
from tqdm import tqdm

from vidore_benchmark.compression.token_pooling import BaseEmbeddingPooler
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever
from vidore_benchmark.utils.iter_utils import batched


class BaseViDoReEvaluator(ABC):
    """
    Base evaluator for the ViDoRe benchmark.
    """

    def __init__(
        self,
        vision_retriever: VisionRetriever,
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

    def _get_query_and_passage_embeddings(
        self,
        ds: Dataset,
        passage_column: str,
        queries: List[str],
        batch_query: int,
        batch_passage: int,
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Get the query and passage embeddings for the given dataset using the retriever used during class
        instance creation.

        Args:
            ds (Dataset): The dataset containing the queries and passages.
            passage_column (str): The column name containing the passages (i.e. images or text).
            queries (List[str]): The list of queries.
            batch_query (int): The batch size for the queries.
            batch_passage (int): The batch size for the passages.

        Returns:
            Tuple[Union[torch.Tensor, List[torch.Tensor]], Union[torch.Tensor, List[torch.Tensor]]]: The query
                and passage embeddings.
        """
        # Get the embeddings for the queries
        query_embeddings = self.vision_retriever.forward_queries(queries, batch_size=batch_query)

        # Get the embeddings for the passages
        passage_embeddings: List[torch.Tensor] = []

        # NOTE: To prevent overloading the RAM for large datasets, we will load the passages (images)
        # that will be fed to the model in batches (this should be fine for queries as their memory footprint
        # is negligible. This optimization is about efficient data loading, and is not related to the model's
        # forward pass which is also batched.

        dataloader_prebatch_size = 10 * batch_passage

        for ds_batch in tqdm(
            batched(ds, n=dataloader_prebatch_size),
            desc="Dataloader pre-batching",
            total=math.ceil(len(ds) / (dataloader_prebatch_size)),
        ):
            passages: List[Any] = [batch[passage_column] for batch in ds_batch]
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
