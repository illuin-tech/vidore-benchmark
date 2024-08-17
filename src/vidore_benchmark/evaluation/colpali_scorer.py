from typing import List

import torch
from mteb.evaluation.evaluators import RetrievalEvaluator

from vidore_benchmark.utils.torch_utils import get_torch_device


class ColPaliScorer:
    """
    Custom scorer for the ColPali retriever.

    NOTE: The `is_multi_vector` parameter was used in the paper for the BiPali model from the
    "ColPali: Efficient Document Retrieval with Vision Language Models" paper. However, BiPali
    is not yet implemented in this repository.
    """

    def __init__(
        self,
        is_multi_vector: bool = False,
        device: str = "auto",
    ):
        self.is_multi_vector = is_multi_vector
        self.mteb_evaluator = RetrievalEvaluator()
        self.device = get_torch_device(device)

    def evaluate(self, qs: List[torch.Tensor], ps: List[torch.Tensor], batch_size: int) -> torch.Tensor:
        if self.is_multi_vector:
            scores = self.evaluate_colbert(qs, ps, batch_size)
        else:
            scores = self.evaluate_biencoder(qs, ps)

        assert scores.shape[0] == len(qs), f"Expected {len(qs)} scores, got {scores.shape[0]}"

        if scores.dtype in [torch.float16, torch.bfloat16]:
            scores = scores.to(torch.float32)
        return scores

    def evaluate_colbert(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        batch_size: int = 128,
    ) -> torch.Tensor:
        """
        Evaluate the similarity scores using the ColBERT scoring function.
        """
        scores: List[torch.Tensor] = []
        for i in range(0, len(qs), batch_size):
            scores_batch: List[torch.Tensor] = []

            qs_batch = torch.nn.utils.rnn.pad_sequence(
                qs[i : i + batch_size],
                batch_first=True,
                padding_value=0,
            ).to(self.device)

            for j in range(0, len(ps), batch_size):
                ps_batch = torch.nn.utils.rnn.pad_sequence(
                    ps[j : j + batch_size],
                    batch_first=True,
                    padding_value=0,
                ).to(self.device)

                if ps_batch.dtype == torch.int8 and qs_batch.dtype == torch.int8:
                    # NOTE: Prevent int8 overflow by casting to int16.
                    scores_batch.append(
                        torch.einsum("bnd,csd->bcns", qs_batch, ps_batch).max(dim=3)[0].sum(dim=2, dtype=torch.int16)
                    )
                else:
                    scores_batch.append(torch.einsum("bnd,csd->bcns", qs_batch, ps_batch).max(dim=3)[0].sum(dim=2))

            scores.append(torch.cat(scores_batch, dim=1).cpu())

        return torch.cat(scores, dim=0)

    def evaluate_biencoder(self, qs: List[torch.Tensor], ps: List[torch.Tensor]) -> torch.Tensor:
        """
        Evaluate the similarity scores using a simple dot product between the query and
        the passage embeddings.
        """
        qs_stacked = torch.stack(qs)
        ps_stacked = torch.stack(ps)
        scores = torch.einsum("bd,cd->bc", qs_stacked, ps_stacked)
        return scores
