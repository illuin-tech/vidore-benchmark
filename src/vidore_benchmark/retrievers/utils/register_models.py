from typing import Any, Dict

VISION_RETRIEVER_REGISTRY: Dict[str, Any] = {}
TEXT_RETRIEVER_REGISTRY: Dict[str, Any] = {}


def register_vision_retriever(model_name: str):
    def decorator(cls):
        print(f"Registering vision retriever {model_name}")
        VISION_RETRIEVER_REGISTRY[model_name] = cls
        return cls

    return decorator


def register_text_retriever(model_name: str):
    def decorator(cls):
        print(f"Registering text retriever {model_name}")
        TEXT_RETRIEVER_REGISTRY[model_name] = cls
        return cls

    return decorator
