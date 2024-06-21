from FlagEmbedding import BGEM3FlagModel
from transformers import AutoModel, AutoImageProcessor

from vidore_benchmark.retrievers.utils.register_models import (
    TEXT_RETRIEVER_REGISTRY,
    VISION_RETRIEVER_REGISTRY,
)
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever


def create_vision_retriever(model_name: str, *args, **kwargs) -> VisionRetriever:
    """
    Create a vision retriever instance based on the model name.
    """

    if model_name in VISION_RETRIEVER_REGISTRY:
        retriever_class = VISION_RETRIEVER_REGISTRY[model_name]
        visual_embedding = True

    elif model_name in TEXT_RETRIEVER_REGISTRY:
        retriever_class = TEXT_RETRIEVER_REGISTRY[model_name]
        visual_embedding = False
    else:
        raise ValueError(f"Unknown model name: {model_name} or model is not initialized correctly.")

    return retriever_class(visual_embedding=visual_embedding)
