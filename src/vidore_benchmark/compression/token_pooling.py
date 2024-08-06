from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch
from scipy.cluster.hierarchy import fcluster, linkage
from vidore_benchmark.utils.torch_utils import get_torch_device


class BaseEmbeddingPooler(ABC):
    """
    Abstract class for pooling embeddings.
    """

    @abstractmethod
    def pool_embeddings(self, p_embeddings: torch.Tensor) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Return the pooled embeddings and the mapping from cluster id to token indices.
        """
        pass


class HierarchicalEmbeddingPooler(BaseEmbeddingPooler):
    """
    Hierarchical pooling of embeddings based on the similarity between tokens.
    """

    def __init__(self, pool_factor: int, device: str = "auto"):
        self.pool_factor = pool_factor
        self.device = get_torch_device(device)

    def pool_embeddings(self, p_embeddings: torch.Tensor) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Return the pooled embeddings and the mapping from cluster id to token indices.

        Input:
        - p_embeddings: tensor of shape (token_length, embedding_dim)
        Output:
        - pooled_embeddings: tensor of shape (num_clusters, embedding_dim)

        NOTE: This method doesn't support batched inputs because:
        - the sequence lengths can be different.
        - scipy doesn't support batched inputs.
        """
        p_embeddings = p_embeddings.to(self.device)
        pooled_embeddings = []
        token_length = p_embeddings.size(0)

        if token_length == 1:
            raise ValueError("The input tensor must have more than one token.")

        similarities = torch.mm(p_embeddings, p_embeddings.t()).to(torch.float32)
        similarities = 1 - similarities.cpu().numpy()

        Z = linkage(similarities, metric="euclidean", method="ward")  # noqa: N806
        max_clusters = max(token_length // self.pool_factor, 1)
        cluster_labels = fcluster(Z, t=max_clusters, criterion="maxclust")

        cluster_id_to_indices: Dict[int, torch.Tensor] = {}

        for cluster_id in range(1, max_clusters + 1):
            cluster_indices = torch.where(torch.tensor(cluster_labels == cluster_id, device=self.device))[0]
            cluster_id_to_indices[cluster_id] = cluster_indices
            if cluster_indices.numel() > 0:
                pooled_embedding = p_embeddings[cluster_indices].mean(dim=0)
                pooled_embeddings.append(pooled_embedding)

        pooled_embeddings = torch.stack(pooled_embeddings, dim=0)
        return pooled_embeddings, cluster_id_to_indices
