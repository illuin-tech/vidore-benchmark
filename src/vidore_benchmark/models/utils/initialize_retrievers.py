from FlagEmbedding import BGEM3FlagModel
from transformers import AutoModel, AutoImageProcessor

from vidore_benchmark.models.utils.register_models import (
    RETRIEVER_COLLATOR_REGISTRY,
    TEXT_RETRIEVER_REGISTRY,
    VISION_RETRIEVER_REGISTRY,
)
from vidore_benchmark.models.vision_retriever import VisionRetriever


def create_vision_retriever(model_name: str, *args, **kwargs) -> VisionRetriever:
    """
    Create a vision retriever instance based on the model name.
    """

    if model_name in VISION_RETRIEVER_REGISTRY:
        retriever_class = VISION_RETRIEVER_REGISTRY[model_name]
        is_vision_retriever = True

    elif model_name in TEXT_RETRIEVER_REGISTRY:
        retriever_class = TEXT_RETRIEVER_REGISTRY[model_name]
        is_vision_retriever = False
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model = initialize_model(model_name, *args, **kwargs)
    processor = initialize_processor(model_name, *args, **kwargs)
    collator = initialize_collator(model_name, *args, **kwargs)

    return retriever_class(model, processor, collator, is_vision_retriever=is_vision_retriever)


def initialize_model(model_name: str, *args, **kwargs):
    if model_name == "BAAI/bge-m3":
        return BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    # Add more models as needed
    if model_name == "jinaai/jina-clip-v1":
        return AutoModel.from_pretrained("jinaai/jina-clip-v1", trust_remote_code=True)
    
    if model_name == "nomic-ai/nomic-embed-vision-v1.5":
        return AutoModel.from_pretrained("nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def initialize_processor(model_name: str, *args, **kwargs):
    if model_name == "BAAI/bge-m3":
        return None  
    # Add more processors as needed
    if model_name == "jinaai/jina-clip-v1":
        return None 
    if model_name == "nomic-ai/nomic-embed-vision-v1.5":
        return AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5")
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def initialize_collator(model_name: str, *args, **kwargs):
    if model_name in RETRIEVER_COLLATOR_REGISTRY:
        return RETRIEVER_COLLATOR_REGISTRY[model_name](*args, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
