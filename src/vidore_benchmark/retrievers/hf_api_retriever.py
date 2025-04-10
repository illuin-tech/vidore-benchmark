from __future__ import annotations

import asyncio
import base64
import os
from io import BytesIO
from typing import List, Optional, Union

import aiohttp
import torch
from dotenv import load_dotenv
from PIL import Image
from tqdm.asyncio import tqdm_asyncio

from vidore_benchmark.retrievers.base_vision_retriever import BaseVisionRetriever
from vidore_benchmark.retrievers.registry_utils import register_vision_retriever
from vidore_benchmark.utils.iter_utils import batched

load_dotenv(override=True)


@register_vision_retriever("hf-endpoint")
class HFEndpointRetriever(BaseVisionRetriever):
    def __init__(self, pretrained_model_name_or_path: str, **kwargs):
        super().__init__(use_visual_embedding=True)
        self.url = pretrained_model_name_or_path
        self.HEADERS = {
            "Accept": "application/json",
            "Authorization": f"Bearer {os.getenv('HF_TOKEN')}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def convert_image_to_base64(image: Image.Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    async def post_images(self, session: aiohttp.ClientSession, encoded_images: List[str]):
        payload = {"inputs": {"images": encoded_images}}
        async with session.post(self.url, headers=self.HEADERS, json=payload) as response:
            return await response.json()

    async def post_queries(self, session: aiohttp.ClientSession, queries: List[str]):
        payload = {"inputs": {"queries": queries}}
        async with session.post(self.url, headers=self.HEADERS, json=payload) as response:
            return await response.json()

    async def call_api_queries(self, queries: List[str]):
        embeddings = []
        semaphore = asyncio.Semaphore(16)
        batch_size = 1
        query_batches = list(batched(queries, batch_size))

        async with aiohttp.ClientSession() as session:

            async def sem_post(batch):
                async with semaphore:
                    return await self.post_queries(session, batch)

            tasks = [asyncio.create_task(sem_post(batch)) for batch in query_batches]

            # ORDER-PRESERVING
            results = await tqdm_asyncio.gather(*tasks, desc="Query batches")

            for result in results:
                embeddings.extend(result.get("embeddings", []))

        return embeddings

    async def call_api_images(self, images_b64: List[str]):
        embeddings = []
        semaphore = asyncio.Semaphore(16)
        batch_size = 1
        image_batches = list(batched(images_b64, batch_size))

        async with aiohttp.ClientSession() as session:

            async def sem_post(batch):
                async with semaphore:
                    return await self.post_images(session, batch)

            tasks = [asyncio.create_task(sem_post(batch)) for batch in image_batches]

            # ORDER-PRESERVING
            results = await tqdm_asyncio.gather(*tasks, desc="Doc batches")

            for result in results:
                embeddings.extend(result.get("embeddings", []))

        return embeddings

    def forward_queries(self, queries: List[str], batch_size: int, **kwargs) -> torch.Tensor:
        response = asyncio.run(self.call_api_queries(queries))
        return response

    def forward_passages(self, passages: List[Image.Image], batch_size: int, **kwargs) -> torch.Tensor:
        response = asyncio.run(self.call_api_images([self.convert_image_to_base64(doc) for doc in passages]))
        return response

    def get_scores(
        self,
        query_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        passage_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        def score_single_vector(
            qs: List[torch.Tensor],
            ps: List[torch.Tensor],
            device: Optional[Union[str, torch.device]] = None,
        ) -> torch.Tensor:
            """
            Compute the dot product score for the given single-vector query and passage embeddings.
            """
            device = "cpu" if device is None else device

            if len(qs) == 0:
                raise ValueError("No queries provided")
            if len(ps) == 0:
                raise ValueError("No passages provided")

            qs_stacked = torch.stack(qs).to(device)
            ps_stacked = torch.stack(ps).to(device)

            scores = torch.einsum("bd,cd->bc", qs_stacked, ps_stacked)
            assert scores.shape[0] == len(qs), f"Expected {len(qs)} scores, got {scores.shape[0]}"

            scores = scores.to(torch.float32)
            return scores

        def score_multi_vector(
            qs: Union[torch.Tensor, List[torch.Tensor]],
            ps: Union[torch.Tensor, List[torch.Tensor]],
            batch_size: int = 128,
            device: Optional[Union[str, torch.device]] = None,
        ) -> torch.Tensor:
            """
            Compute the late-interaction/MaxSim score (ColBERT-like) for the given multi-vector
            query embeddings (`qs`) and passage embeddings (`ps`). For ColPali, a passage is the
            image of a document page.

            Because the embedding tensors are multi-vector and can thus have different shapes, they
            should be fed as:
            (1) a list of tensors, where the i-th tensor is of shape (sequence_length_i, embedding_dim)
            (2) a single tensor of shape (n_passages, max_sequence_length, embedding_dim) -> usually
                obtained by padding the list of tensors.

            Args:
                qs (`Union[torch.Tensor, List[torch.Tensor]`): Query embeddings.
                ps (`Union[torch.Tensor, List[torch.Tensor]`): Passage embeddings.
                batch_size (`int`, *optional*, defaults to 128): Batch size for computing scores.
                device (`Union[str, torch.device]`, *optional*): Device to use for computation. If not
                    provided, uses `get_torch_device("auto")`.

            Returns:
                `torch.Tensor`: A tensor of shape `(n_queries, n_passages)` containing the scores. The score
                tensor is saved on the "cpu" device.
            """
            device = "cpu" if device is None else device
            batch_size = batch_size or 128

            if len(qs) == 0:
                raise ValueError("No queries provided")
            if len(ps) == 0:
                raise ValueError("No passages provided")

            scores_list: List[torch.Tensor] = []

            for i in range(0, len(qs), batch_size):
                scores_batch = []
                qs_batch = torch.nn.utils.rnn.pad_sequence(
                    qs[i : i + batch_size], batch_first=True, padding_value=0
                ).to(device)
                for j in range(0, len(ps), batch_size):
                    ps_batch = torch.nn.utils.rnn.pad_sequence(
                        ps[j : j + batch_size], batch_first=True, padding_value=0
                    ).to(device)
                    scores_batch.append(torch.einsum("bnd,csd->bcns", qs_batch, ps_batch).max(dim=3)[0].sum(dim=2))
                scores_batch = torch.cat(scores_batch, dim=1).cpu()
                scores_list.append(scores_batch)

            scores = torch.cat(scores_list, dim=0)
            assert scores.shape[0] == len(qs), f"Expected {len(qs)} scores, got {scores.shape[0]}"

            scores = scores.to(torch.float32)
            return scores

        try:
            return score_multi_vector(query_embeddings, passage_embeddings, batch_size=batch_size)
        except Exception:
            return score_single_vector(query_embeddings, passage_embeddings)
