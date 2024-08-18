import logging
from typing import Dict, Type

from vidore_benchmark.retrievers.vision_retriever import VisionRetriever

VISION_RETRIEVER_REGISTRY: Dict[str, Type[VisionRetriever]] = {}

logger = logging.getLogger(__name__)


def register_vision_retriever(model_name: str):
    def decorator(cls):
        VISION_RETRIEVER_REGISTRY[model_name] = cls
        logger.debug("Registered vision retriever `%s`", model_name)
        return cls

    return decorator
