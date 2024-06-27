# How to implement my own VisionRetriever?

- Instantiate a class inherited from the `VisionRetriever` abstract class.
- Implement the `forward_query`, `forward_documents` and `get_scores` abstract methods.
- [Optional] Implement your custom metric logic in `get_relevant_docs_results` and `compute_metrics`.
- Add decorator `@register_vision_retriever({{my_vision_retriever}})`
- Import your class to the `vidore_benchmark/retrievers/__init__.py` file

You can look at the [`DummyRegister`](https://github.com/tonywu71/vidore-benchmark/blob/main/src/vidore_benchmark/retrievers/dummy_retriever.py) for a simple example.
