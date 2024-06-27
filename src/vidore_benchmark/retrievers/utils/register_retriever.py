from typing import Dict

from vidore_benchmark.retrievers.vision_retriever import VisionRetriever

VISION_RETRIEVER_REGISTRY: Dict[str, VisionRetriever] = {}


def register_vision_retriever(model_name: str):
    def decorator(cls):
        VISION_RETRIEVER_REGISTRY[model_name] = cls
        return cls

    return decorator
