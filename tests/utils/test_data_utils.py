from datasets import Dataset

from vidore_benchmark.utils.data_utils import deduplicate_dataset_rows


def test_deduplicate_dataset_rows():
    # Test normal deduplication
    ds = Dataset.from_dict(
        {
            "query": ["q1", "q2", "q1", None, "q3", "q2", None],
            "other_col": [1, 2, 3, 4, 5, 6, 7],
        }
    )

    deduped_ds = deduplicate_dataset_rows(ds, "query")

    assert len(deduped_ds) == 3  # Should only have q1, q2, q3
    assert list(deduped_ds["query"]) == ["q1", "q2", "q3"]
    assert list(deduped_ds["other_col"]) == [1, 2, 5]

    # Test dataset with all None values
    ds_all_none = Dataset.from_dict(
        {
            "query": [None, None, None],
            "other_col": [1, 2, 3],
        }
    )
    deduped_all_none = deduplicate_dataset_rows(ds_all_none, "query")
    assert len(deduped_all_none) == 0
