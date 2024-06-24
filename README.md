# vidore-benchmark
[[Blog]]()
[[Paper]]()
[[ColPali Model card]]()
[[ViDoRe Dataset card]]()
[[Colab example]]()
[[HuggingFace Space]]()

Vision Document Retrieval (ViDoRe): Benchmark

## Evaluate a retriever on ViDoRE
To evaluate a retriever on ViDoRe benchmark, execute the following function: 

```bash
python scripts/evaluate_retriever.py --collection-name vidore/vidore-benchmark-667173f98e70a1c0fa4db00d --split 'test' --model-name [hf_model_name]
```

alternatively, you can evaluate your model on a single dataset 

```bash
python scripts/evaluate_retriever.py --dataset-name [hf_dataset_name] --model-name [hf_model_name]
```

Supported visual retrievers are ColPali (ours), JinaClip, Nomic Vision and SigLip. If you want to add more please refer to the section below [add link]()


## Implement your own retriever

Here are the steps to create your custom model for retrieval evaluation.

- Instantiate a class inherited from `VisionRetriever` abstract class. 
- Implement `forward_query`, `forward_documents` and `get_scores` abstract methods. 
- [OPTIONAL] Implement `get_relevant_docs_results` and `compute_metrics` if you want custom implementation for metric computation.

If you want to call directly your class by your model_name (e.g. in  `evaluate.py` script) you can do the following:

- Add decorator `@register_vision_retriever([your model name])`
- Import your class to the `vidore_benchmark/retrievers/__init__.py` file

Examples be found in `src/vidore_benchmark/retrievers/` :

```python
#src/vidore_benchmark/retrievers/dummy_retriever.py
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever
from vidore_benchmark.retrievers.utils.register_models import register_text_retriever
from typing import List
from tqdm import tqdm
from PIL import Image
import torch

@register_vision_retriever("dummy_retriever")
class DummyRetriever(VisionRetriever):
    def __init__(
        self,
        emb_dim_query: int = 512,
        emb_dim_doc: int = 512,
    ):
        super().__init__()
        self.emb_dim_query = emb_dim_query
        self.emb_dim_doc = emb_dim_doc

    @property
    def use_visual_embedding(self) -> bool:
        return False

    def forward_queries(self, queries: List[str], **kwargs) -> torch.Tensor:
        return torch.randn(len(queries), self.emb_dim_query)

    def forward_documents(self, documents: List[Image.Image], **kwargs) -> torch.Tensor:
        return torch.randn(len(documents), self.emb_dim_doc)

    def get_scores(
        self,
        queries: List[str],
        documents: List[Image.Image | str],
        batch_query: int,
        batch_doc: int,
        **kwargs,
    ) -> torch.Tensor:
        scores = torch.randn(len(queries), len(documents))
        return scores

```

## Baselines Reproducibility

We compared ColPali to existing strong baseline for document retrieving, close to what is done in industry. We chunk the different document using widely used open-source tool [unstructured](unstructured.io) and treat the visual chunks (figures and tables) with OCR or Captioning. Below are the commands to reproduce baselines results (also works with `--model-name bm25`). 
- OCR : 
```bash
python scripts/evaluate.py --split 'test' --collection-name vidore/vidore-chunk-ocr-baseline-666acce88c294ef415548a56 --model-name BAAI/bge-m3
```
- Captioning :  
```bash
python scripts/evaluate.py --split 'test' --collection-name vidore/vidore-captioning-baseline-6658a2a62d857c7a345195fd  --model-name BAAI/bge-m3
```

## Associated Paper

**ColPali: Efficient Document Retrieval with Vision Language Models**
Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, CELINE HUDELOT, Pierre Colombo
