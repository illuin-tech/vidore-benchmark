from FlagEmbedding import BGEM3FlagModel
from transformers import AutoModel, AutoImageProcessor

from vidore_benchmark.retrievers.utils.register_models import (
    RETRIEVER_COLLATOR_REGISTRY,
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
        text_only = False

    elif model_name in TEXT_RETRIEVER_REGISTRY:
        retriever_class = TEXT_RETRIEVER_REGISTRY[model_name]
        text_only = True
    else:
        raise ValueError(f"Unknown model name: {model_name} or model is not initialized correctly.")

    return retriever_class(text_only=text_only )


def initialize_model(model_name: str, *args, **kwargs):
    
    if model_name == "nomic-ai/nomic-embed-vision-v1.5":
        return AutoModel.from_pretrained("nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def initialize_processor(model_name: str, *args, **kwargs):
    
    if model_name == "nomic-ai/nomic-embed-vision-v1.5":
        return AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5")
    else:
        raise ValueError(f"Unknown model name: {model_name}")