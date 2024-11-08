from typing import Dict, List, Tuple, Union

import torch
from colpali_engine.trainer.eval_utils import CustomRetrievalEvaluator
from datasets import Dataset


def score_multi_vector(
    emb_queries: Union[torch.Tensor, List[torch.Tensor]],
    emb_passages: Union[torch.Tensor, List[torch.Tensor]],
    batch_size: int,
) -> torch.Tensor:
    """
    Evaluate the similarity scores using the MaxSim scoring function.

    Inputs:
        - emb_queries: List of query embeddings, each of shape (n_seq, emb_dim).
        - emb_passages: List of document embeddings, each of shape (n_seq, emb_dim).
        - batch_size: Batch size for the similarity computation.
    """
    if len(emb_queries) == 0:
        raise ValueError("No queries provided")
    if len(emb_passages) == 0:
        raise ValueError("No passages provided")

    if emb_queries[0].device != emb_passages[0].device:
        raise ValueError("Queries and passages must be on the same device")

    if emb_queries[0].dtype != emb_passages[0].dtype:
        raise ValueError("Queries and passages must have the same dtype")

    scores: List[torch.Tensor] = []

    for i in range(0, len(emb_queries), batch_size):
        batch_scores = []
        qs_batch = torch.nn.utils.rnn.pad_sequence(
            emb_queries[i : i + batch_size],
            batch_first=True,
            padding_value=0,
        )
        for j in range(0, len(emb_passages), batch_size):
            ps_batch = torch.nn.utils.rnn.pad_sequence(
                emb_passages[j : j + batch_size],
                batch_first=True,
                padding_value=0,
            )
            batch_scores.append(torch.einsum("bnd,csd->bcns", qs_batch, ps_batch).max(dim=3)[0].sum(dim=2))
        batch_scores = torch.cat(batch_scores, dim=1)
        scores.append(batch_scores)

    return torch.cat(scores, dim=0)


def get_relevant_docs_results(
    ds: Dataset,
    queries: List[str],
    scores: torch.Tensor,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Get the relevant passages and the results from the scores.

    Outputs:
    - relevant_docs: Dict[str, float]
    {
        "query_0": {"doc_0": 1},
        "query_1": {"doc_1": 1},
        ...
    }
    - results: Dict[str, Dict[str, float]] with shape:
    {
        "query_0": {"doc_i": 19.125, "doc_1": 18.75, ...},
        "query_1": {"doc_j": 17.25, "doc_1": 16.75, ...},
        ...
    }
    """
    relevant_docs = {}
    results = {}

    query_to_filename = {query: image_filename for query, image_filename in zip(ds["query"], ds["image_filename"])}
    passage_to_filename = {docidx: image_filename for docidx, image_filename in enumerate(ds["image_filename"])}

    for query, score_per_query in zip(queries, scores):
        relevant_docs[query] = {query_to_filename[query]: 1}

        for docidx, score in enumerate(score_per_query):
            filename = passage_to_filename[docidx]
            score_passage = float(score.item())

            if query in results:
                current_score = results[query].get(filename, 0)
                results[query][filename] = max(current_score, score_passage)
            else:
                results[query] = {filename: score_passage}

    return relevant_docs, results


def compute_retrieval_metrics(
    relevant_docs: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    **kwargs,
) -> Dict[str, float]:
    """
    Compute the MTEB metrics for retrieval.
    """

    mteb_evaluator = CustomRetrievalEvaluator()

    ndcg, _map, recall, precision, naucs = mteb_evaluator.evaluate(
        qrels=relevant_docs,
        results=results,
        k_values=mteb_evaluator.k_values,
        ignore_identical_ids=kwargs.get("ignore_identical_ids", True),
    )

    mrr = mteb_evaluator.evaluate_custom(relevant_docs, results, mteb_evaluator.k_values, "mrr")

    scores = {
        **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
        **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
        **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
        **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
        **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr[0].items()},
        **{f"naucs_at_{k.split('@')[1]}": v for (k, v) in naucs.items()},
    }

    return scores
