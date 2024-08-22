from abc import ABC, abstractmethod

import torch


class BaseEmbeddingQuantizer(ABC):
    """
    Abstract class for embedding quantization.
    """

    @abstractmethod
    def quantize(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Quantize the input embeddings.

        Input:
        - embeddings (torch.Tensor): input embeddings (batch_size, seq_len, embedding_dim)
        Output:
        - quantized_embeddings (torch.Tensor): quantized embeddings (batch_size, seq_len, quantized_dim)
        """
        pass
