import warnings
from typing import Optional

import numpy as np
import torch
from sentence_transformers import quantize_embeddings

from vidore_benchmark.compression.quantization import BaseEmbeddingQuantizer


class EmbeddingInt8Quantizer(BaseEmbeddingQuantizer):
    """
    Embedding quantizer that quantizes the embeddings to int8.
    """

    def __init__(
        self,
        ranges: Optional[np.ndarray] = None,
        calibration_embeddings: Optional[np.ndarray] = None,
    ):
        self.ranges = ranges
        self.calibration_embeddings = calibration_embeddings

    def set_ranges(self, ranges: np.ndarray) -> None:
        """
        Set the min/max ranges for quantization.

        Args:
        - ranges (np.ndarray): min/max ranges (2, embedding_dim).
        """
        self.ranges = ranges

    def set_calibration_embeddings(self, calibration_embeddings: np.ndarray) -> None:
        """
        Set the calibration embeddings for quantization.

        Args:
        - calibration_embeddings (np.ndarray): calibration embeddings (n_examples, embedding_dim).
        """
        self.calibration_embeddings = calibration_embeddings

    def quantize(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Quantize the input embeddings to 8-bit integers.

        Input:
        - embeddings (torch.Tensor): input embeddings (batch_size, seq_len, embedding_dim)
        Output:
        - quantized_embeddings (torch.Tensor): quantized embeddings (batch_size, seq_len, quantized_dim)
        """

        warnings.warn(
            "EmbeddingInt8Quantizer is experimental and may change in the future.",
            UserWarning,
        )

        if self.ranges is None and self.calibration_embeddings is None:
            raise ValueError("Ranges and calibration embeddings must be set before quantizing embeddings.")
        if self.ranges is not None and self.calibration_embeddings is not None:
            raise ValueError(
                "Only one of ranges and calibration embeddings should be set before quantizing embeddings."
            )

        emb_dim = embeddings.shape[-1]
        emb_quantized = quantize_embeddings(
            embeddings.to(torch.float16).reshape(-1, emb_dim),
            precision="int8",
            ranges=self.ranges,
            calibration_embeddings=self.calibration_embeddings,
        ).reshape(embeddings.shape)

        return torch.tensor(emb_quantized, device=embeddings.device, dtype=torch.int8)
