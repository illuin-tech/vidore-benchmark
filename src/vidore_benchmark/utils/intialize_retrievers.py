from FlagEmbedding import BGEM3FlagModel

from vidore_benchmark.models.vision_retriever import VisionRetriever
from vidore_benchmark.utils.registration import RETRIEVER_COLLATOR_REGISTRY, VISION_RETRIEVER_REGISTRY


def create_vision_retriever(model_name: str, *args, **kwargs) -> VisionRetriever:
    """
    Create a vision retriever instance based on the model name.
    """

    if model_name not in VISION_RETRIEVER_REGISTRY:
        raise ValueError(f"Model {model_name} is not registered in the vision retriever registry.")

    retriever_class = VISION_RETRIEVER_REGISTRY[model_name]

    model = initialize_model(model_name, *args, **kwargs)
    processor = initialize_processor(model_name, *args, **kwargs)
    collator = initialize_collator(model_name, *args, **kwargs)

    return retriever_class(model, processor, collator)


def initialize_model(model_name: str, *args, **kwargs):
    if model_name == "BGEM3":
        return BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    # Add more models as needed
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def initialize_processor(model_name: str, *args, **kwargs):
    if model_name == "BGEM3":
        return None  # Placeholder, update if necessary
    # Add more processors as needed
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def initialize_collator(model_name: str, *args, **kwargs):
    if model_name in RETRIEVER_COLLATOR_REGISTRY:
        return RETRIEVER_COLLATOR_REGISTRY[model_name](*args, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
