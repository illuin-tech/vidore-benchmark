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
        if not qs:
            raise ValueError("No queries provided.")
        if not ps:
            raise ValueError("No passages provided.")

        if qs[0].dtype != ps[0].dtype:
            raise ValueError("Queries and passages must have the same dtype.")

        is_dtype_int8 = ps[0].dtype == torch.int8 and qs[0].dtype == torch.int8

        for i in range(0, len(qs), batch_size):
            scores_batch: List[torch.Tensor] = []

            qs_batch = torch.nn.utils.rnn.pad_sequence(
                qs[i : i + batch_size],
                batch_first=True,
                padding_value=0,
            )

            for j in range(0, len(ps), batch_size):
                ps_batch = torch.nn.utils.rnn.pad_sequence(
                    ps[j : j + batch_size],
                    batch_first=True,
                    padding_value=0,
                )

                if is_dtype_int8:
                    # NOTE: CUDA does not support int8 operations, so we need to move the
                    # tensors to the CPU for the MaxSim operation.
                    scores_batch.append(
                        torch.einsum(
                            "bnd,csd->bcns",
                            qs_batch.to("cpu"),
                            ps_batch.to("cpu"),
                        )
                        .max(dim=3)[0]
                        .sum(dim=2, dtype=torch.int16)  # prevent int8 overflow
                        .to(self.device)
                    )
                else:
                    scores_batch.append(
                        torch.einsum(
                            "bnd,csd->bcns",
                            qs_batch.to(self.device),
                            ps_batch.to(self.device),
                        )
                        .max(dim=3)[0]
                        .sum(dim=2)
                    )

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
