from __future__ import annotations

import logging
import math
from typing import List, Optional, Union

import torch
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from vidore_benchmark.retrievers.registry_utils import register_vision_retriever
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever
from vidore_benchmark.utils.iter_utils import batched
from vidore_benchmark.utils.torch_utils import get_torch_device

logger = logging.getLogger(__name__)


load_dotenv(override=True)


@register_vision_retriever("dse-qwen2")
class DSEQwen2Retriever(VisionRetriever):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "MrLight/dse-qwen2-2b-mrl-v1",
        num_image_tokens: int = 1024,  # 2560 is the original value
        device: str = "auto",
    ):
        super().__init__()

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            raise ImportError(
                'Install the missing dependencies with `pip install "vidore-benchmark[dse]"` '
                "to use DSEQwen2Retriever."
            )

        self.device = get_torch_device(device)
        logger.info(f"Using device: {self.device}")

        min_pixels = 1 * 28 * 28
        max_pixels = num_image_tokens * 28 * 28

        self.processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path, min_pixels=min_pixels, max_pixels=max_pixels
        )
        self.process_vision_info = process_vision_info

        self.model = (
            Qwen2VLForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path,
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
                torch_dtype=torch.bfloat16,
            )
            .to(self.device)
            .eval()
        )
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

    def forward_queries(
        self,
        queries: List[str],
        batch_size: int,
        **kwargs,
    ) -> List[torch.Tensor]:
        qs = []

        for batch_query in tqdm(
            batched(queries, batch_size),
            desc="Forwarding query batches",
            total=math.ceil(len(queries) / batch_size),
            leave=False,
        ):
            query_messages = []

            for query in batch_query:
                message = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": Image.new("RGB", (28, 28)),
                                "resized_height": 1,
                                "resized_width": 1,
                            },
                            # need a dummy image here for an easier process.
                            {"type": "text", "text": f"Query: {query}"},
                        ],
                    }
                ]
                query_messages.append(message)

            query_texts = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) + "<|endoftext|>"
                for msg in query_messages
            ]
            query_image_inputs, query_video_inputs = self.process_vision_info(query_messages)

            query_inputs = self.processor(
                text=query_texts,
                images=query_image_inputs,
                videos=query_video_inputs,
                padding="longest",
                return_tensors="pt",
            ).to(self.device)

            cache_position = torch.arange(0, len(query_texts))
            query_inputs = self.model.prepare_inputs_for_generation(
                **query_inputs, cache_position=cache_position, use_cache=False
            )

            with torch.no_grad():
                output = self.model(**query_inputs, return_dict=True, output_hidden_states=True)
            query_embeddings = self.get_embedding(output.hidden_states[-1], 1536)

            qs.extend(list(torch.unbind(query_embeddings.to("cpu"))))

        return qs

    def forward_passages(
        self,
        passages: List[Image.Image],
        batch_size: int,
        **kwargs,
    ) -> List[torch.Tensor]:
        ds = []

        for batch_doc in tqdm(
            batched(passages, batch_size),
            desc="Forwarding passage batches",
            total=math.ceil(len(passages) / batch_size),
            leave=False,
        ):
            doc_messages = []

            for doc in batch_doc:
                message = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": doc},
                            # 'resized_height':680 , 'resized_width':680} # adjust the image sizes
                            {"type": "text", "text": "What is shown in this image?"},
                        ],
                    }
                ]
                doc_messages.append(message)

            doc_texts = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) + "<|endoftext|>"
                for msg in doc_messages
            ]

            doc_image_inputs, doc_video_inputs = self.process_vision_info(doc_messages)
            doc_inputs = self.processor(
                text=doc_texts, images=doc_image_inputs, videos=doc_video_inputs, padding="longest", return_tensors="pt"
            ).to(self.device)

            cache_position = torch.arange(0, len(doc_texts))
            doc_inputs = self.model.prepare_inputs_for_generation(
                **doc_inputs, cache_position=cache_position, use_cache=False
            )

            with torch.no_grad():
                output = self.model(**doc_inputs, return_dict=True, output_hidden_states=True)
            doc_embeddings = self.get_embedding(output.hidden_states[-1], 1536)

            ds.extend(list(torch.unbind(doc_embeddings.to("cpu"))))

        return ds

    def get_scores(
        self,
        query_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        passage_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Dot-product similarity between queries and passages.
        """
        if isinstance(query_embeddings, list):
            query_embeddings = torch.stack(query_embeddings)
        if isinstance(passage_embeddings, list):
            passage_embeddings = torch.stack(passage_embeddings)

        scores = torch.einsum("bd,cd->bc", query_embeddings, passage_embeddings)

        return scores
