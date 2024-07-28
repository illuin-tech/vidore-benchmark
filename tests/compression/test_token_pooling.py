import pytest
import torch
from vidore_benchmark.compression.token_pooling import HierarchicalEmbeddingPooler


@pytest.fixture
def sample_embeddings() -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )


def test_hierarchical_embedding_pooler_initialization():
    pooler = HierarchicalEmbeddingPooler(pool_factor=2, device="cpu")
    assert pooler.pool_factor == 2
    assert str(pooler.device) == "cpu"


def test_hierarchical_embedding_pooler_output_shape(sample_embeddings: torch.Tensor):
    pooler = HierarchicalEmbeddingPooler(pool_factor=2, device="cpu")
    pooled_embeddings, cluster_id_to_indices = pooler.pool_embeddings(sample_embeddings)

    assert isinstance(pooled_embeddings, torch.Tensor)
    assert pooled_embeddings.shape[1] == sample_embeddings.shape[1]
    assert pooled_embeddings.shape[0] <= len(cluster_id_to_indices)


def test_hierarchical_embedding_pooler_output_values(sample_embeddings: torch.Tensor):
    pooler = HierarchicalEmbeddingPooler(pool_factor=2, device="cpu")
    pooled_embeddings, cluster_id_to_indices = pooler.pool_embeddings(sample_embeddings)

    expected_pooled_embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    expected_cluster_id_to_indices = {
        1: torch.tensor([0, 3, 4, 5]),
        2: torch.tensor([1]),
        3: torch.tensor([2]),
    }

    assert torch.allclose(pooled_embeddings, expected_pooled_embeddings)
    assert all(
        [
            torch.allclose(cluster_id_to_indices[cluster_id], expected_cluster_indices)
            for cluster_id, expected_cluster_indices in expected_cluster_id_to_indices.items()
        ]
    )


def test_hierarchical_embedding_pooler_with_different_pool_factors(sample_embeddings: torch.Tensor):
    for pool_factor in [1, 2, 3]:
        pooler = HierarchicalEmbeddingPooler(pool_factor=pool_factor, device="cpu")
        pooled_embeddings, cluster_id_to_indices = pooler.pool_embeddings(sample_embeddings)
        expected_num_clusters = (sample_embeddings.shape[0] + pool_factor - 1) // pool_factor
        assert pooled_embeddings.shape[0] <= expected_num_clusters
        assert pooled_embeddings.shape[0] <= len(cluster_id_to_indices)


def test_hierarchical_embedding_pooler_should_raise_error_with_single_token():
    single_token_embeddings = torch.rand(1, 768)
    pooler = HierarchicalEmbeddingPooler(pool_factor=2, device="cpu")

    with pytest.raises(ValueError):
        pooler.pool_embeddings(single_token_embeddings)


def test_hierarchical_embedding_pooler_large_input():
    large_embeddings = torch.rand(1000, 768)

    pooler = HierarchicalEmbeddingPooler(pool_factor=10, device="cpu")
    pooled_embeddings, cluster_id_to_indices = pooler.pool_embeddings(large_embeddings)

    assert pooled_embeddings.shape[0] < large_embeddings.shape[0]
    assert pooled_embeddings.shape[0] <= len(cluster_id_to_indices)
