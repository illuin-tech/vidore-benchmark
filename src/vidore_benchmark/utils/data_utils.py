from typing import List, TypeVar

from datasets import Dataset as HfDataset
from torch.utils.data import Dataset as TorchDataset

T = TypeVar("T")


class ListDataset(TorchDataset[T]):
    def __init__(self, elements: List[T]):
        self.elements = elements

    def __len__(self) -> int:
        return len(self.elements)

    def __getitem__(self, idx: int) -> T:
        return self.elements[idx]


def deduplicate_dataset_rows(ds: HfDataset, target_column: str) -> HfDataset:
    """
    Remove duplicate rows from a dataset based on values in a target column.

    Args:
        ds (Dataset): The dataset to deduplicate.
        target_column (str): The column to use for deduplication.

    Returns:
        Dataset: The deduplicated dataset.
    """
    if target_column not in ds.column_names:
        raise ValueError(f"Column '{target_column}' not found in dataset.")

    seen_values = set()
    keep_mask = []

    for value in ds[target_column]:
        if value is None:
            keep_mask.append(False)
            continue

        if value not in seen_values:
            seen_values.add(value)
            keep_mask.append(True)
        else:
            keep_mask.append(False)

    return ds.select([i for i, keep in enumerate(keep_mask) if keep])
