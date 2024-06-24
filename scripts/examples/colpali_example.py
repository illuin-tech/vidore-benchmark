import torch
from abc import abstractmethod
from typing import List

from vidore_benchmark.retrievers.vision_retriever import VisionRetriever
from PIL import Image
from transformers import AutoProcessor
from torch.utils.data import DataLoader
from tqdm import tqdm

from typing import cast

from datasets import Dataset, load_dataset
from dotenv import load_dotenv

# make sure to install the custom_colbert package
from custom_colbert.models.paligemma_colbert_architecture import ColPali
from custom_colbert.trainer.retrieval_evaluator import CustomEvaluator as ColPaliScorer
from vidore_benchmark.evaluation.evaluate import evaluate_dataset

load_dotenv(override=True)


class ColPaliRetriever(VisionRetriever):
    """
    Abstract class for ViDoRe retrievers.
    """

    def __init__(self):
        super().__init__()
        model_name = "coldoc/colpali-3b-mix-448"
        self.model = ColPali.from_pretrained("google/paligemma-3b-mix-448", torch_dtype=torch.bfloat16,
                                             device_map="cuda").eval()
        self.model.load_adapter(model_name)
        self.scorer = ColPaliScorer(is_multi_vector=True)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.device = self.model.device

    @property
    def use_visual_embedding(self) -> bool:
        """
        The child class should instantiate the `use_visual_embedding` property:
        - True if the retriever uses native visual embeddings (e.g. JINA-Clip, ColPali)
        - False if the retriever uses text embeddings and possibly VLM-generated captions (e.g. BM25).
        """
        return True

    def process_images(self, images, **kwargs):
        texts_doc = ["Describe the image."] * len(images)
        images = [image.convert("RGB") for image in images]

        batch_doc = self.processor(
            text=texts_doc,
            images=images,
            return_tensors="pt",
            padding="longest",
            max_length=kwargs.get("max_length", 50) + self.processor.image_seq_length,
        )
        return batch_doc

    def process_queries(self, queries, **kwargs) -> torch.Tensor:
        """
        Forward pass the processed queries.
        """
        texts_query = []
        for query in queries:
            query = f"Question: {query}<unused0><unused0><unused0><unused0><unused0>"
            texts_query.append(query)

        mock_image = Image.new("RGB", (448, 448), (255, 255, 255))
        batch_query = self.processor(
            images=[mock_image.convert("RGB")] * len(texts_query),
            # NOTE: the image is not used in batch_query but it is required for calling the processor
            text=texts_query,
            return_tensors="pt",
            padding="longest",
            max_length=kwargs.get("max_length", 50) + self.processor.image_seq_length,
        )
        del batch_query["pixel_values"]

        batch_query["input_ids"] = batch_query["input_ids"][..., self.processor.image_seq_length:]
        batch_query["attention_mask"] = batch_query["attention_mask"][..., self.processor.image_seq_length:]
        return batch_query

    @abstractmethod
    def forward_queries(self, queries, **kwargs) -> List[torch.Tensor]:
        """
        Forward pass the processed queries.
        """
        # run inference - docs
        dataloader = DataLoader(
            queries,
            batch_size=kwargs.get("bs", 4),
            shuffle=False,
            collate_fn=self.process_queries,
        )
        qs = []
        for batch_query in tqdm(dataloader):
            with torch.no_grad():
                batch_query = {k: v.to(self.device) for k, v in batch_query.items()}
                embeddings_query = self.model(**batch_query)
                qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))

        return qs

    @abstractmethod
    def forward_documents(self, documents, **kwargs) -> List[torch.Tensor]:
        """
        Forward pass the processed documents (i.e. page images).
        """

        # run inference - docs
        dataloader = DataLoader(
            documents,
            batch_size=kwargs.get("bs", 4),
            shuffle=False,
            collate_fn=lambda x: self.process_images(x),
        )
        ds = []
        for batch_doc in tqdm(dataloader):
            with torch.no_grad():
                batch_doc = {k: v.to(self.device) for k, v in batch_doc.items()}
                embeddings_doc = self.model(**batch_doc)
            ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))
        return ds

    @abstractmethod
    def get_scores(
            self,
            queries: List[str],
            documents: List["Image.Image | str"],
            batch_query: int,
            batch_doc: int,
            **kwargs,
    ) -> torch.Tensor:
        """
        Get the similarity scores between queries and documents.
        """
        qs = self.forward_queries(queries, bs=batch_query)
        ds = self.forward_documents(documents, bs=batch_doc)

        scores = self.scorer.evaluate(qs, ds)
        return scores


def main():
    """
    Debugging script
    """
    my_retriever = ColPaliRetriever()

    dataset = cast(Dataset, load_dataset("coldoc/shiftproject_test", split="test"))

    print("Dataset loaded")
    metrics = evaluate_dataset(my_retriever, dataset, batch_query=4, batch_doc=4)  # type: ignore

    print(metrics)


if __name__ == "__main__":
    main()
