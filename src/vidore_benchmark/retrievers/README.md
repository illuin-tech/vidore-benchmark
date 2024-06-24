# Add a new retriever model 

Here are the steps to create your custom model for retrieval evaluation.

- Instantiate a class inherited from `VisionRetriever` abstract class. 
- Implement `forward_query`, `forward_documents` and `get_scores` abstract methods. 
- [OPTIONAL] Implement `get_relevant_docs_results` and `compute_metrics` if you want custom implementation for metric computation.


If you want to call directly your class by your model_name (e.g. in  `evaluate.py` script) you can do the following:

- Add decorator `@register_vision_retriever([your model name])` or `@register_text_retriever([your model name])` depending on whether your embedding model support images.
- Import your class to the `vidore_benchmark/retrievers/__init__.py` file


