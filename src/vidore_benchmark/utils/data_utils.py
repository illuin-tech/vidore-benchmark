import logging
import os
from pathlib import Path
from typing import List, TypeVar

import huggingface_hub
from datasets import Dataset as HfDataset
from datasets import get_dataset_config_names
from torch.utils.data import Dataset as TorchDataset

T = TypeVar("T")

logger = logging.getLogger(__name__)


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


def get_datasets_from_collection(collection_name: str) -> List[str]:
    """
    Get dataset names from a local directory or a HuggingFace collection.

    Args:
        collection_name: Local dirpath or HuggingFace collection ID

    Returns:
        List of dataset names
    """
    if Path(collection_name).is_dir():
        logger.info(f"Loading datasets from local directory: `{collection_name}`")
        dataset_names = os.listdir(collection_name)
        dataset_names = [os.path.join(collection_name, dataset) for dataset in dataset_names]
    else:
        logger.info(f'Loading datasets from the Hf Hub collection: "{collection_name}"')
        collection = huggingface_hub.get_collection(collection_name)
        dataset_names = [dataset_item.item_id for dataset_item in collection.items]
    return dataset_names

def get_configs_to_evaluate(
    dataset: str, language_configs: List[str], exclude_default: bool = False
) -> List[str]:
    """
    Get the list of language configurations to evaluate for a given dataset. The returned list will either
    be the intersection of the available configurations and the provided language configurations, or if there
    is only a 'default' available config for the dataset, a list of [None] will be returned.

    Args:
        dataset: Dataset name
        language_configs: List of language configurations to evaluate
        exclude_default: Exclude the default configuration

    Returns:
        List of configurations to evaluate
    """
    try:
        available_configs = get_dataset_config_names(dataset)
        if available_configs == ["default"] and not exclude_default:
            logging.info(f"Dataset has no language configs, loading default: {dataset}")
            return [None]
        elif language_configs:
            configs = list(set(available_configs) & set(language_configs))
            if not configs:
                logging.info(
                    f"No matching language configs for dataset {dataset}. Available: {available_configs}"
                )
                return []
            return configs
        else:
            return available_configs
    except Exception as e:
        logging.error(f"Error getting configurations for dataset {dataset}: {str(e)}")
        return []
