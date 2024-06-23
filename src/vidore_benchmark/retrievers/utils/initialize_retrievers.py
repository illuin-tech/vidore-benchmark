from vidore_benchmark.retrievers.utils.register_models import VISION_RETRIEVER_REGISTRY
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever


def load_vision_retriever_from_registry(model_name: str, *args, **kwargs) -> VisionRetriever:
    """
    Create a vision retriever instance based on the model name.
    """

    if model_name in VISION_RETRIEVER_REGISTRY:
        retriever_class = VISION_RETRIEVER_REGISTRY[model_name]
    else:
        raise ValueError(
            f"Unknown model name `{model_name}`. Available models: {list(VISION_RETRIEVER_REGISTRY.keys())}"
        )

    return retriever_class()
