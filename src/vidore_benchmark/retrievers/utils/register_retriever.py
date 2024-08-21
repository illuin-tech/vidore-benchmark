from typing import Dict, Type

from loguru import logger

from vidore_benchmark.retrievers.vision_retriever import VisionRetriever

VISION_RETRIEVER_REGISTRY: Dict[str, Type[VisionRetriever]] = {}


def register_vision_retriever(model_name: str):
    def decorator(cls):
        VISION_RETRIEVER_REGISTRY[model_name] = cls

        # NOTE: To see this log, use `logger.enable("vidore_benchmark")` at the very top of the script.
        logger.debug("Registered vision retriever `{}`", model_name)

        return cls

    return decorator
