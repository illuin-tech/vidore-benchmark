import math

import pytest
import torch

from vidore_benchmark.compression.quantization.embedding_binarizer import EmbeddingBinarizer


@pytest.fixture
def embedding_binarizer():
    return EmbeddingBinarizer()


def test_embedding_binarizer_initialization():
    binarizer = EmbeddingBinarizer()
    assert isinstance(binarizer, EmbeddingBinarizer)


def test_pad_last_dim_to_multiple_of_8_on_2d_vector():
    binarizer = EmbeddingBinarizer()

    # Test when padding is not needed
    x = torch.randn(2, 8)
    padded = binarizer.pad_last_dim_to_multiple_of_8(x)
    assert padded.shape == (2, 8)
    assert torch.all(padded == x)

    # Test when padding is needed
    x = torch.randn(2, 5)
    padded = binarizer.pad_last_dim_to_multiple_of_8(x)
    assert padded.shape == (2, 8)
    assert torch.all(padded[:, :5] == x)
    assert torch.all(padded[:, 5:] == 0)


def test_pad_last_dim_to_multiple_of_8_on_3d_vector():
    binarizer = EmbeddingBinarizer()

    # Test when padding is not needed
    x = torch.randn(2, 3, 8)
    padded = binarizer.pad_last_dim_to_multiple_of_8(x)
    assert padded.shape == (2, 3, 8)
    assert torch.all(padded == x)

    # Test when padding is needed
    x = torch.randn(2, 3, 5)
    padded = binarizer.pad_last_dim_to_multiple_of_8(x)
    assert padded.shape == (2, 3, 8)
    assert torch.all(padded[:, :, :5] == x)
    assert torch.all(padded[:, :, 5:] == 0)


def test_quantize_batched_embeddings(embedding_binarizer):
    # Test with embeddings that don't need padding
    embeddings = torch.randn(2, 3, 8)
    quantized = embedding_binarizer.quantize(embeddings)
    assert quantized.shape == (2, 3, 1)
    assert quantized.dtype == torch.int8

    # Test with embeddings that need padding
    embeddings = torch.randn(2, 3, 5)
    quantized = embedding_binarizer.quantize(embeddings)
    assert quantized.shape == (2, 3, 1)
    assert quantized.dtype == torch.int8


def test_quantize_values(embedding_binarizer):
    embeddings = torch.tensor(
        [
            [
                [1.0, 2.0, -1.0, -1.0],  # should get thresholded to [1, 1, 0, 0, 0, 0, 0, 0]
            ]
        ],
        dtype=torch.float32,
        device="cpu",
    )

    quantized = embedding_binarizer.quantize(embeddings)

    expected = torch.tensor([[[64]]])
    assert torch.all(quantized == expected)


@pytest.mark.parametrize(
    "batch_size,seq_length,dim",
    [
        (1, 1, 8),
        (2, 3, 16),
        (5, 10, 64),
        (3, 5, 100),
    ],
)
def test_quantize_various_shapes(embedding_binarizer, batch_size, seq_length, dim):
    embeddings = torch.randn(batch_size, seq_length, dim)
    quantized = embedding_binarizer.quantize(embeddings)
    expected_dim = math.ceil(dim / 8)
    assert quantized.shape == (batch_size, seq_length, expected_dim)


def test_quantize_device_consistency(embedding_binarizer):
    embeddings = torch.randn(2, 3, 8)
    quantized_cpu = embedding_binarizer.quantize(embeddings.cpu())
    assert quantized_cpu.device == torch.device("cpu")
