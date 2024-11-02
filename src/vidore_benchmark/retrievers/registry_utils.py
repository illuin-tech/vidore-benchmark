import logging
from typing import Dict, Optional, Type

from vidore_benchmark.retrievers.vision_retriever import VisionRetriever

VISION_RETRIEVER_REGISTRY: Dict[str, Type[VisionRetriever]] = {}

logger = logging.getLogger(__name__)


def register_vision_retriever(model_class: str):
    def decorator(cls):
        VISION_RETRIEVER_REGISTRY[model_class] = cls

        # NOTE: To see this log, use `logger.enable("vidore_benchmark")` at the very top of the script.
        logger.debug("Registered vision retriever `{}`", model_class)

        return cls

    return decorator


def load_vision_retriever_class_from_registry(model_class: str) -> Type[VisionRetriever]:
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
) -> VisionRetriever:
    """
    Create a vision retriever class instance.
    If `model_name` is provided, the retriever will be instantiated with the given model name or path.
    """

    retriever_class = load_vision_retriever_class_from_registry(model_class)

    if pretrained_model_name_or_path is not None:
        retriever = retriever_class(pretrained_model_name_or_path=pretrained_model_name_or_path)
    else:
        retriever = retriever_class()

    return retriever
