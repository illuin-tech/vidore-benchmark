from typing import List

import torch


def get_colbert_similarity(
    emb_queries: List[torch.Tensor],
    emb_documents: List[torch.Tensor],
    batch_size: int,
) -> torch.Tensor:
    """
    Evaluate the similarity scores using the ColBERT scoring function.

    Inputs:
        - emb_queries: List of query embeddings, each of shape (n_seq, emb_dim).
        - emb_documents: List of document embeddings, each of shape (n_seq, emb_dim).
        - batch_size: Batch size for the similarity computation.
    """
    if not emb_queries or not emb_documents:
        return torch.tensor([])

    if emb_queries[0].device != emb_documents[0].device:
        raise ValueError("The device of the query and document embeddings must be the same.")

    scores = []
    for i in range(0, len(emb_queries), batch_size):
        scores_batch = []
        qs_batch = torch.nn.utils.rnn.pad_sequence(
            emb_queries[i : i + batch_size],
            batch_first=True,
            padding_value=0,
        )
        for j in range(0, len(emb_documents), batch_size):
            ps_batch = torch.nn.utils.rnn.pad_sequence(
                emb_documents[j : j + batch_size],
                batch_first=True,
                padding_value=0,
            )
            scores_batch.append(torch.einsum("bnd,csd->bcns", qs_batch, ps_batch).max(dim=3)[0].sum(dim=2))
        scores_batch = torch.cat(scores_batch, dim=1)
        scores.append(scores_batch)
    scores = torch.cat(scores, dim=0)
    return scores
