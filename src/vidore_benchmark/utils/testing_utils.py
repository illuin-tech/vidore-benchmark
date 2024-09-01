import gc

import torch


def tear_down_torch():
    """
    Teardown for PyTorch.
    Should be used after each torch-based vision retriever.
    """
    gc.collect()
    torch.cuda.empty_cache()
