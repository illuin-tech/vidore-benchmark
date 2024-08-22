import math
import warnings

import torch
from sentence_transformers import quantize_embeddings

from vidore_benchmark.compression.quantization import BaseEmbeddingQuantizer


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

        Input:
        - embeddings (torch.Tensor): input embeddings (batch_size, seq_len, embedding_dim)
        Output:
        - quantized_embeddings (torch.Tensor): quantized embeddings (batch_size, seq_len, quantized_dim)
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
