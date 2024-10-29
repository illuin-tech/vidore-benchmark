from typing import Optional, Type

from vidore_benchmark.retrievers.utils.register_retriever import VISION_RETRIEVER_REGISTRY
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever


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

    retriever = retriever_class(pretrained_model_name_or_path=pretrained_model_name_or_path)

    return retriever
