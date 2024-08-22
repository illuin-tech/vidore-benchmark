import math
import warnings
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
from sentence_transformers import quantize_embeddings


class BaseEmbeddingQuantizer(ABC):
    """
    Abstract class for embedding quantization.
    """

    @abstractmethod
    def quantize(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Quantize the input embeddings.
        """
        pass


class EmbeddingBinarizer(BaseEmbeddingQuantizer):
    """
    Embedding quantizer that binarizes the embeddings and packs them into 8-bit integers.
    """

    @staticmethod
    def pad_last_dim_to_multiple_of_8(x: torch.Tensor) -> torch.Tensor:
        """
        Pad the last dimension of the tensor to be a multiple of 8.
        """
        last_dim = x.shape[-1]
        new_last_dim = math.ceil(last_dim / 8) * 8
        padding = [0, new_last_dim - last_dim]
        padded_tensor = torch.nn.functional.pad(x, padding)
        return padded_tensor

    def quantize(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Quantize the input embeddings using binary quantization:
        1. each element is thresholded to 0 if it is less than 0, and 1 otherwise.
        2. the thresholded embeddings are packed into 8-bit integers.
        """
        warnings.warn(
            "EmbeddingBinarizer is experimental and may change in the future.",
            UserWarning,
        )

        # NOTE: We pad the last dimension of the embeddings to be a multiple of 8
        # because the quantize_embeddings function takes a 2D tensor as input and pack
        # the bits into 8-bit integers (int8). To make our 3D tensors compatible, we
        # pad the last dimension to be a multiple of 8 and reshape the tensor to 2D.
        # This ensures that each int8 only contains bits from a single token.
        emb_padded = self.pad_last_dim_to_multiple_of_8(embeddings)

        batch_size, *intermediate_dims, dim = emb_padded.shape
        assert dim % 8 == 0, "The last dimension of the embeddings should be a multiple of 8."
        packed_dim = dim // 8

        emb_binarized = quantize_embeddings(
            emb_padded.to(torch.float16).reshape(batch_size, -1),
            precision="binary",
        ).reshape(batch_size, *intermediate_dims, packed_dim)

        return torch.tensor(emb_binarized, device=embeddings.device, dtype=torch.int8)


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

        batch_size, emb_dim = embeddings.shape
        emb_quantized = quantize_embeddings(
            embeddings.to(torch.float16).reshape(-1, emb_dim),
            precision="int8",
            ranges=self.ranges,
            calibration_embeddings=self.calibration_embeddings,
        ).reshape(embeddings.shape)

        return torch.tensor(emb_quantized, device=embeddings.device, dtype=torch.int8)
