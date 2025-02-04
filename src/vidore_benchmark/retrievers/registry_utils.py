import logging
from typing import Dict, Optional, Type

from vidore_benchmark.retrievers.base_vision_retriever import BaseVisionRetriever

VISION_RETRIEVER_REGISTRY: Dict[str, Type[BaseVisionRetriever]] = {}

logger = logging.getLogger(__name__)


def register_vision_retriever(model_class: str):
    def decorator(cls):
        VISION_RETRIEVER_REGISTRY[model_class] = cls
        logger.debug("Registered vision retriever `{}`", model_class)
        return cls

    return decorator


def load_vision_retriever_class_from_registry(model_class: str) -> Type[BaseVisionRetriever]:
    """
    Get a vision retriever class.

    To name an instance of VisionRetriever, use the following decorator:
    >>> @register_vision_retriever("my_vision_retriever")
    >>> class MyVisionRetriever(VisionRetriever):
    >>> ...
    """

    if model_class in VISION_RETRIEVER_REGISTRY:
        retriever_class = VISION_RETRIEVER_REGISTRY[model_class]
    else:
        raise ValueError(
            f"Unknown model name `{model_class}`. Available models: {list(VISION_RETRIEVER_REGISTRY.keys())}"
        )
    return retriever_class


def load_vision_retriever_from_registry(
    model_class: str,
    pretrained_model_name_or_path: Optional[str] = None,
    **kwargs,
) -> BaseVisionRetriever:
    """
    Create a vision retriever class instance.
    """

    retriever_class = load_vision_retriever_class_from_registry(model_class)

    if pretrained_model_name_or_path:
        retriever = retriever_class(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            **kwargs,
        )
    else:
        retriever = retriever_class(**kwargs)

    return retriever
