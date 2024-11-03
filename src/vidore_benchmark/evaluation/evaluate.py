from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union

import torch
from datasets import Dataset
from tqdm import tqdm

from vidore_benchmark.compression.token_pooling import BaseEmbeddingPooler
from vidore_benchmark.retrievers.bm25_retriever import BM25Retriever
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever
from vidore_benchmark.utils.iter_utils import batched


def evaluate_dataset(
    vision_retriever: VisionRetriever,
    ds: Dataset,
    batch_query: int,
    batch_passage: int,
    batch_score: Optional[int] = None,
    embedding_pooler: Optional[BaseEmbeddingPooler] = None,
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

    # Dataset: sanity check
    passage_column_name = "image" if vision_retriever.use_visual_embedding else "text_description"
    required_columns = ["query", passage_column_name, "image_filename"]

    if not all(col in ds.column_names for col in required_columns):
        raise ValueError(f"Dataset should contain the following columns: {required_columns}")

    # Remove `None` queries (i.e. pages for which no question was generated) and duplicates
    # queries = list(set(ds["query"]))
    # --> old buggy behavior - this differs from colpali-engine implementation where duplicates are NOT removed
    # for fairness with externally evaluated retrievers since bug, we maintain this behavior and remove duplicates
    # This slightly boosts scores on docvqa typically
    seen_queries = set()
    queries = []
    for query in ds["query"]:
        if query is not None and query not in seen_queries:
            queries.append(query)
            seen_queries.add(query)

    if len(queries) == 0:
        raise ValueError("All queries are None")

    # Edge case: using the BM25Retriever
    if isinstance(vision_retriever, BM25Retriever):
        passages = ds[passage_column_name]
        scores = vision_retriever.get_scores_bm25(queries=queries, passages=passages)
        relevant_docs, results = vision_retriever.get_relevant_docs_results(ds, queries, scores)
        metrics = vision_retriever.compute_metrics(relevant_docs, results)
        return metrics

    # Get the embeddings for the queries and passages
    emb_queries = vision_retriever.forward_queries(queries, batch_size=batch_query)

    # NOTE: To prevent overloading the RAM for large datasets, we will load the passages (images)
    # that will be fed to the model in batches (this should be fine for queries as their memory footprint
    # is negligible. This optimization is about efficient data loading, and is not related to the model's
    # forward pass which is also batched.
    emb_passages: List[torch.Tensor] = []

    dataloader_prebatch_size = 10 * batch_passage

    for passage_batch in tqdm(
        batched(ds, n=dataloader_prebatch_size),
        desc="Dataloader pre-batching",
        total=math.ceil(len(ds) / (dataloader_prebatch_size)),
    ):
        passages: List[Any] = [db[passage_column_name] for db in passage_batch]
        batch_emb_passages = vision_retriever.forward_passages(passages, batch_size=batch_passage)
        if isinstance(batch_emb_passages, torch.Tensor):
            batch_emb_passages = list(torch.unbind(batch_emb_passages))
            emb_passages.extend(batch_emb_passages)
        else:
            emb_passages.extend(batch_emb_passages)

    if embedding_pooler is not None:
        for idx, emb_document in tqdm(enumerate(emb_passages), total=len(emb_passages), desc="Pooling embeddings..."):
            emb_document, _ = embedding_pooler.pool_embeddings(emb_document)
            emb_passages[idx] = emb_document

    # Get the similarity scores
    scores = vision_retriever.get_scores(emb_queries, emb_passages, batch_size=batch_score)

    # Get the relevant passages and results
    relevant_docs, results = vision_retriever.get_relevant_docs_results(ds, queries, scores)

    # Compute the MTEB metrics
    metrics = vision_retriever.compute_metrics(relevant_docs, results)

    return metrics


def get_top_k(
    vision_retriever: VisionRetriever,
    queries: List[str],
    emb_queries: Union[torch.Tensor, List[torch.Tensor]],
    emb_passages: Union[torch.Tensor, List[torch.Tensor]],
    file_names: List[str],
    k: int,
    batch_score: Optional[int] = None,
) -> Dict[str, OrderedDict[str, float]]:
    """
    Get the top-k passages for a given query.

    Output:
    {
        query_1: {
            passage_1: score_1,
            ...
        },
        ...
    }
    """
    scores = vision_retriever.get_scores(emb_queries, emb_passages, batch_size=batch_score)
    passages2filename = {doc_idx: image_filename for doc_idx, image_filename in enumerate(file_names)}
    query_to_score = {}

    for query, score_per_query in zip(queries, scores):
        for docidx, score in enumerate(score_per_query):
            filename = passages2filename[docidx]
            score_passage = float(score.item())

            if query in query_to_score:
                query_to_score[query][filename] = max(query_to_score[query].get(filename, 0), score_passage)
            else:
                query_to_score[query] = {filename: score_passage}

    # Get the top-k passages
    for query in query_to_score:
        query_to_score[query] = OrderedDict(
            sorted(query_to_score[query].items(), key=lambda item: item[1], reverse=True)[:k]
        )

    return query_to_score
