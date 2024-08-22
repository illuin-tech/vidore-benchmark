import numpy as np
import pytest
import torch
from vidore_benchmark.compression.quantization.embedding_int8_quantizer import EmbeddingInt8Quantizer

EMBEDDING_DIM = 8
RANGES = np.array([[-1.0, 1.0]] * EMBEDDING_DIM).T


@pytest.fixture
def embedding_quantizer():
    return EmbeddingInt8Quantizer(ranges=RANGES)


def test_embedding_quantizer_initialization():
    quantizer = EmbeddingInt8Quantizer(ranges=RANGES)
    assert isinstance(quantizer, EmbeddingInt8Quantizer)


def test_quantize_batched_embeddings(embedding_quantizer):
    embeddings = torch.randn(2, 3, 8)
    quantized = embedding_quantizer.quantize(embeddings)
    assert quantized.shape == embeddings.shape
    assert quantized.dtype == torch.int8


def test_quantize_values():
    embeddings = torch.tensor(
        [
            [
                [1.0, 0.0, -1.0, -1.0],
            ]
        ],
        dtype=torch.float32,
        device="cpu",
    )

    embeddings_dim = embeddings.shape[-1]
    naive_ranges = np.array([[-1.0, 1.0]] * embeddings_dim).T
    embedding_quantizer = EmbeddingInt8Quantizer(ranges=naive_ranges)

    quantized = embedding_quantizer.quantize(embeddings)

    expected = torch.tensor([[[127, 0, -128, -128]]], dtype=torch.int8, device="cpu")
    assert torch.all(quantized == expected)


def test_quantize_device_consistency(embedding_quantizer):
    embeddings = torch.randn(2, 3, 8)
    quantized_cpu = embedding_quantizer.quantize(embeddings.cpu())
    assert quantized_cpu.device == torch.device("cpu")
