# How to implement my own VisionRetriever?

To implement your own VisionRetriever, you need to:

1. Instantiate a class inherited from the `VisionRetriever` abstract class.
2. Implement the `forward_query`, `forward_documents` and `get_scores` abstract methods.
3. Add decorator `@register_vision_retriever({my_vision_retriever})` to register your class in the retriever registry.
4. Import your class to the `vidore_benchmark/retrievers/__init__.py` file.

You can look at the [`dummy_vision_retriever.py`](./dummy_vision_retriever.py) file for a simple example.
