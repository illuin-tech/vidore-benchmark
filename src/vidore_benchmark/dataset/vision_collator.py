from abc import ABC, abstractmethod
from typing import Dict, List

import torch


class VisionCollator(ABC):
    """
    Abstract class for ViDoRe collators.
    """

    col_query: str = "query"
    col_document: str = "document"

    @abstractmethod
    def __call__(self, batch: Dict[str, List[Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of data.

        The output batch should contain the following keys:
        - "query": a tensor of shape (batch_size, ...)
        - "document": a tensor of shape (batch_size, ...)
        """
        pass
