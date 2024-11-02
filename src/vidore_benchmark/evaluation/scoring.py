from typing import List, Union

import torch


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
