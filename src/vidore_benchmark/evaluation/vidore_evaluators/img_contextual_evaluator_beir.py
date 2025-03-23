from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, TypedDict, Union

import torch
from datasets import Dataset
from tqdm import tqdm

from vidore_benchmark.evaluation.vidore_evaluators.base_vidore_evaluator import BaseViDoReEvaluator
from vidore_benchmark.retrievers.base_vision_retriever import BaseVisionRetriever
from vidore_benchmark.utils.iter_utils import batched

logger = logging.getLogger(__name__)


class BEIRDatasetWithDocs(TypedDict):
    corpus: Dataset
    queries: Dataset
    qrels: Dataset


class ImageContextEvaluatorBEIR(BaseViDoReEvaluator):
    def __init__(
        self,
        vision_retriever: BaseVisionRetriever,
        corpus_id_column: Optional[str] = None,
        query_id_column: Optional[str] = None,
        query_column: Optional[str] = None,
        passage_column: Optional[str] = None,
        score_column: Optional[str] = None,
        prev_image_column: Optional[str] = None,
        next_image_column: Optional[str] = None,
    ):
        super().__init__(vision_retriever=vision_retriever)

        self.corpus_id_column = corpus_id_column if corpus_id_column else "corpus-id"
        self.query_id_column = query_id_column if query_id_column else "query-id"
        self.query_column = query_column if query_column else "query"
        if passage_column:
            self.passage_column = passage_column
        else:
            self.passage_column = "image" if self.vision_retriever.use_visual_embedding else "text_description"
        self.score_column = score_column if score_column else "score"
        self.prev_image_column = prev_image_column if prev_image_column else "prev-image"
        self.next_image_column = next_image_column if next_image_column else "next-image"

    def evaluate_dataset(
        self,
        ds: BEIRDatasetWithDocs,
        batch_query: int,
        batch_passage: int,
        batch_score: Optional[int] = None,
        dataloader_prebatch_query: Optional[int] = None,
        dataloader_prebatch_passage: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Optional[float]]:
        ds_corpus = ds["corpus"]
        ds_queries = ds["queries"]
        ds_qrels = ds["qrels"]

        passage_ids: List[str] = [str(elt) for elt in ds_corpus[self.corpus_id_column]]
        query_ids: List[str] = [str(elt) for elt in ds_queries[self.query_id_column]]

        qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
        for qrel in ds_qrels:
            query_id = str(qrel[self.query_id_column])
            corpus_id = str(qrel[self.corpus_id_column])
            qrels[query_id][corpus_id] = qrel[self.score_column]

        query_embeddings = self._get_query_embeddings(
            ds=ds_queries,
            query_column=self.query_column,
            batch_query=batch_query,
            dataloader_prebatch_size=dataloader_prebatch_query,
        )
        passage_embeddings = self._get_passage_embeddings(
            ds_corpus=ds_corpus,
            passage_column=self.passage_column,
            prev_image_column=self.prev_image_column,
            next_image_column=self.next_image_column,
            batch_passage=batch_passage,
            dataloader_prebatch_size=dataloader_prebatch_passage,
        )

        scores = self.vision_retriever.get_scores(
            query_embeddings=query_embeddings,
            passage_embeddings=passage_embeddings,
            batch_size=batch_score,
        )

        results = self._get_retrieval_results(
            query_ids=query_ids,
            passage_ids=passage_ids,
            scores=scores,
        )

        metrics = self.compute_retrieval_scores(
            qrels=qrels,
            results=results,
            ignore_identical_ids=False,
        )

        return metrics

    def _get_passage_embeddings(
        self,
        ds_corpus: Dataset,
        passage_column: str,
        prev_image_column: str,
        next_image_column: str,
        batch_passage: int,
        dataloader_prebatch_size: Optional[int] = None,
        **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        passage_embeddings: List[torch.Tensor] = []

        if dataloader_prebatch_size is None:
            dataloader_prebatch_size = 10 * batch_passage
        if dataloader_prebatch_size < batch_passage:
            logger.warning(
                f"`dataloader_prebatch_size` ({dataloader_prebatch_size}) is smaller than `batch_passage` "
                f"({batch_passage}). Setting the pre-batch size to the passager batch size."
            )
            dataloader_prebatch_size = batch_passage

        for ds_batch in tqdm(
            batched(ds_corpus, n=dataloader_prebatch_size),
            desc="Dataloader pre-batching for passages",
            total=math.ceil(len(ds_corpus) / (dataloader_prebatch_size)),
        ):
            passages: List[Any] = [batch[passage_column] for batch in ds_batch]
            prev_passages: List[Any] = [batch[prev_image_column] for batch in ds_batch]
            next_passages: List[Any] = [batch[next_image_column] for batch in ds_batch]

            batch_embedding_passages = self.vision_retriever.forward_passages(
                passages=passages,
                prev_passages=prev_passages,
                next_passages=next_passages,
                batch_size=batch_passage,
            )

            if isinstance(batch_embedding_passages, torch.Tensor):
                batch_embedding_passages = list(torch.unbind(batch_embedding_passages.to("cpu")))
                passage_embeddings.extend(batch_embedding_passages)
            else:
                for embedding_passage in batch_embedding_passages:
                    passage_embeddings.append(embedding_passage.to("cpu"))

        return passage_embeddings

    def _get_retrieval_results(
        self,
        query_ids: List[str],
        passage_ids: List[str],
        scores: torch.Tensor,
    ) -> Dict[str, Dict[str, float]]:
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
