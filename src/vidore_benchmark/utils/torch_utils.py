"""
Utility functions for interpretability.
"""

import torch


def get_torch_device(device: str = "auto") -> str:
    """
    Returns the device (string) to be used by PyTorch.

    Defaults to "auto" which will use:
    - "cuda:0" if available
    - else "mps" if available
    - "cpu" otherwise.
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
