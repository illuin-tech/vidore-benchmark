"""
Utility functions for interpretability.
"""

import torch


def get_torch_device(device: str = "auto") -> str:
    """
    Returns the device and dtype to be used for torch tensors.
    """
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        elif torch.backends.mps.is_available():  # for Apple Silicon
            return "mps"
        else:
            return "cpu"
    else:
        return device
