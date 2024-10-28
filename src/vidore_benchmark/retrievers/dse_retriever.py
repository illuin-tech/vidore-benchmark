from __future__ import annotations

import math
from typing import ClassVar, List, Optional, TypeVar

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from vidore_benchmark.utils.iter_utils import batched

try:
    from qwen_vl_utils import process_vision_info
except:
    print("qwen_vl_utils not found")

from dotenv import load_dotenv
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from vidore_benchmark.retrievers.utils.register_retriever import register_vision_retriever
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever
from vidore_benchmark.utils.torch_utils import get_torch_device

T = TypeVar("T")
load_dotenv(override=True)


class ListDataset(Dataset[T]):
    def __init__(self, elements: List[T]):
        self.elements = elements

    def __len__(self) -> int:
        return len(self.elements)

    def __getitem__(self, idx: int) -> T:
        return self.elements[idx]


@register_vision_retriever("dse-qwen2")
class DSERetriever(VisionRetriever):
    """
    ColPali Retriever that implements the model from "ColPali: Efficient Document Retrieval
    with Vision Language Models".
    """

    emb_dim_query: ClassVar[int] = 128
    emb_dim_doc: ClassVar[int] = 128

    def __init__(
        self,
        model_name: str = "MrLight/dse-qwen2-2b-mrl-v1", # "vidore/colqwen-v0.1-merged",
        device: str = "auto",
    ):
        super().__init__()

        self.device = get_torch_device(device)
        logger.info(f"Using device: {self.device}")

        min_pixels = 1 * 28 * 28
        max_pixels = 1024 * 28 * 28 # modified from 2560

        self.processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels,
                                                  max_pixels=max_pixels)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_name,
                                                                attn_implementation="flash_attention_2",
                                                                torch_dtype=torch.bfloat16).to(self.device).eval()
        self.processor.tokenizer.padding_side = "left"
        self.model.padding_side = "left"

        print("Loaded custom processor.\n")


    def get_embedding(self, last_hidden_state: torch.Tensor, dimension: int) -> torch.Tensor:
        reps = last_hidden_state[:, -1]
        reps = torch.nn.functional.normalize(reps[:, :dimension], p=2, dim=-1)
        return reps

    @property
    def use_visual_embedding(self) -> bool:
        return True

    def forward_queries(self, queries: List[str], batch_size: int, **kwargs) -> List[torch.Tensor]:
        qs = []
        for batch_query in tqdm(batched(queries, batch_size),  desc="Query batch",
                                total=math.ceil(len(queries) / batch_size)):
            query_messages = []
            for query in batch_query:
                message = [
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'image', 'image': Image.new('RGB', (28, 28)), 'resized_height': 1,
                             'resized_width': 1},
                            # need a dummy image here for an easier process.
                            {'type': 'text', 'text': f'Query: {query}'},
                        ]
                    }
                ]
                query_messages.append(message)
            query_texts = [
                self.processor.apply_chat_template(msg, tokenize=False,
                                                   add_generation_prompt=True) + "<|endoftext|>"
                for msg in query_messages
            ]
            query_image_inputs, query_video_inputs = process_vision_info(query_messages)
            query_inputs = self.processor(text=query_texts, images=query_image_inputs, videos=query_video_inputs,
                                          padding='longest', return_tensors='pt').to('cuda:0')
            cache_position = torch.arange(0, len(query_texts))
            query_inputs = self.model.prepare_inputs_for_generation(**query_inputs, cache_position=cache_position,
                                                                    use_cache=False)
            with torch.no_grad():
                output = self.model(**query_inputs, return_dict=True, output_hidden_states=True)
            query_embeddings = self.get_embedding(output.hidden_states[-1], 1536)
            qs.extend(list(torch.unbind(query_embeddings.to("cpu"))))

        return qs



    def forward_documents(self, documents: List[Image.Image], batch_size: int, **kwargs) -> List[torch.Tensor]:
        ds = []
        for batch_doc in tqdm(
            batched(documents, batch_size), desc="Document batch", total=math.ceil(len(documents) / batch_size)
        ):
            doc_messages = []
            for doc in batch_doc:
                message = [
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'image', 'image': doc},
                            # 'resized_height':680 , 'resized_width':680} # adjust the image sizes
                            {'type': 'text', 'text': 'What is shown in this image?'}
                        ]
                    }
                ]
                doc_messages.append(message)
            doc_texts = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) + "<|endoftext|>"
                for msg in doc_messages
            ]
            doc_image_inputs, doc_video_inputs = process_vision_info(doc_messages)
            doc_inputs = self.processor(text=doc_texts, images=doc_image_inputs, videos=doc_video_inputs, padding='longest',
                                   return_tensors='pt').to('cuda:0')
            cache_position = torch.arange(0, len(doc_texts))
            doc_inputs = self.model.prepare_inputs_for_generation(**doc_inputs, cache_position=cache_position,
                                                             use_cache=False)
            with torch.no_grad():
                output = self.model(**doc_inputs, return_dict=True, output_hidden_states=True)
            doc_embeddings = self.get_embedding(output.hidden_states[-1], 1536)

            ds.extend(list(torch.unbind(doc_embeddings.to("cpu"))))
        return ds

    def get_scores(
        self,
        list_emb_queries: List[torch.Tensor],
        list_emb_documents: List[torch.Tensor],
        batch_size: Optional[int] = 128,
    ) -> torch.Tensor:
        if batch_size is None:
            raise ValueError("`batch_size` must be provided for ColPaliRetriever's scoring")
        # compute cosine similarity
        qs_stacked = torch.stack(list_emb_queries).to(self.device)
        ps_stacked = torch.stack(list_emb_documents).to(self.device)

        scores = torch.einsum("bd,cd->bc", qs_stacked, ps_stacked)
        scores = scores.to(torch.float32).cpu()

        return scores