from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel


class MetadataModel(BaseModel):
    """
    Metadata data for the ViDoRe benchmark results.

    Example usage:

    ```python
    >>> from datetime import datetime
    >>> metadata = MetadataModel(
            timestamp=datetime.now(),
            vidore_benchmark_version="0.0.1.dev7+g462dc4f.d20241102",
        )
    >>> # Serialize the model to JSON
    >>> json_data = metadata.model_dump_json()
    ```
    """

    timestamp: datetime
    vidore_benchmark_version: str

    # NOTE: `Config` is used to allow extra fields in the model
    class Config:
        extra = "allow"


class ViDoReBenchmarkResults(BaseModel):
    """
    ViDoRe benchmark results.

    This Pydantic model contains:
    - Metadata: Information about the benchmark run, including the timestamp and the `vidore-benchmark` package version.
    - Metrics: Dictionary of metrics: the keys are the dataset names and the values are the metrics for that dataset.

    Example usage:

    ```python
    >>> root = RootModel(
            metadata=Metadata(
                timestamp=datetime.now(),
                vidore_benchmark_version="0.0.1.dev7+g462dc4f.d20241102",
            ),
            metrics={
                "vidore/syntheticDocQA_dummy": {
                    "ndcg_at_1": 1.0,
                    "ndcg_at_3": 1.0,
                }
            }
        )
    >>> # Serialize the model to JSON
    >>> json_data = root.model_dump_json()
    ```
    """

    metadata: MetadataModel
    metrics: Dict[str, Dict[str, Optional[float]]]

    @classmethod
    def merge(cls, results: List["ViDoReBenchmarkResults"]) -> "ViDoReBenchmarkResults":
        """
        Merge multiple `ViDoReBenchmarkResults` instances into a single one.

        Uses the latest timestamp from the input results and combines all metrics.
        If there are conflicting metrics for the same benchmark, the last one in the list takes precedence.

        Args:
            results: List of ViDoReBenchmarkResults to merge

        Returns:
            A new `ViDoReBenchmarkResults` instance containing the merged data

        Raises:
            ValueError: If the input list is empty
        """
        if not results:
            raise ValueError("Cannot merge an empty list of results")

        # Use the metadata from the result with the latest timestamp
        latest_result = max(results, key=lambda x: x.metadata.timestamp)
        merged_metadata = latest_result.metadata

        # Merge all metrics, later results override earlier ones for the same benchmark
        merged_metrics: Dict[str, Dict[str, Optional[float]]] = {}
        for result in results:
            if set(merged_metrics.keys()).intersection(set(result.metrics.keys())):
                raise ValueError("Duplicate dataset keys found in the input results")
            merged_metrics.update(result.metrics)

        return cls(metadata=merged_metadata, metrics=merged_metrics)
