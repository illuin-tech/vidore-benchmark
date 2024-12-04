import json
from datetime import datetime

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from vidore_benchmark.evaluation.eval_manager import EvalManager
from vidore_benchmark.evaluation.interfaces import MetadataModel, ViDoReBenchmarkResults


class TestEvalManagerBase:
    @pytest.fixture
    def sample_data(self):
        return {
            "model1": pd.Series(
                {
                    ("dataset1", "metric1"): 0.8,
                    ("dataset1", "metric2"): 0.7,
                    ("dataset2", "metric1"): 0.6,
                }
            ),
            "model2": pd.Series(
                {
                    ("dataset1", "metric1"): 0.9,
                    ("dataset1", "metric2"): 0.85,
                    ("dataset2", "metric1"): 0.75,
                }
            ),
        }

    @pytest.fixture
    def sample_json_path(self, tmp_path, sample_data):
        data = {"dataset1": {"metric1": 0.8, "metric2": 0.7}, "dataset2": {"metric1": 0.6}}
        json_path = tmp_path / "model1.json"
        with open(json_path, "w") as f:
            json.dump(data, f)
        return json_path

    @pytest.fixture
    def eval_manager(self, sample_data):
        return EvalManager.from_dict(sample_data)


class TestEvalManagerInitialization(TestEvalManagerBase):
    def test_init_empty(self):
        eval_manager = EvalManager()
        assert eval_manager.data.empty

    def test_from_dict(self, sample_data):
        eval_manager = EvalManager.from_dict(sample_data)
        assert not eval_manager.data.empty
        assert list(eval_manager.models) == ["model1", "model2"]
        assert list(eval_manager.datasets) == ["dataset1", "dataset2"]

    def test_from_json(self, sample_json_path):
        eval_manager = EvalManager.from_json(sample_json_path)
        assert not eval_manager.data.empty
        assert list(eval_manager.models) == ["model1"]

    def test_from_multiple_json(self, tmp_path, sample_data):
        data1 = {"dataset1": {"metric1": 0.8, "metric2": 0.7}}
        data2 = {"dataset1": {"metric1": 0.9, "metric2": 0.85}}

        path1 = tmp_path / "model1.json"
        path2 = tmp_path / "model2.json"

        with open(path1, "w") as f:
            json.dump(data1, f)
        with open(path2, "w") as f:
            json.dump(data2, f)

        eval_manager = EvalManager.from_multiple_json([path1, path2])
        assert len(eval_manager.models) == 2

    def test_from_dir(self, tmp_path, sample_data):
        data1 = {"dataset1": {"metric1": 0.8, "metric2": 0.7}}
        data2 = {"dataset1": {"metric1": 0.9, "metric2": 0.85}}

        path1 = tmp_path / "model1.json"
        path2 = tmp_path / "model2.json"

        with open(path1, "w") as f:
            json.dump(data1, f)
        with open(path2, "w") as f:
            json.dump(data2, f)

        eval_manager = EvalManager.from_dir(tmp_path)
        assert len(eval_manager.models) == 2

    def test_from_vidore_results(self):
        results = ViDoReBenchmarkResults(
            metadata=MetadataModel(
                timestamp=datetime.now(),
                vidore_benchmark_version="0.0.1.dev7+g462dc4f.d20241102",
            ),
            metrics={
                "dataset1": {"ndcg_at_1": 0.8, "ndcg_at_3": 0.7},
                "dataset2": {"ndcg_at_1": 0.9, "ndcg_at_3": 0.85},
            },
        )
        eval_manager = EvalManager.from_vidore_results(results, model_name="test_model")
        assert not eval_manager.data.empty
        assert set(eval_manager.datasets) == {"dataset1", "dataset2"}


class TestEvalManagerDataAccess(TestEvalManagerBase):
    def test_property_accessors(self, eval_manager):
        assert list(eval_manager.models) == ["model1", "model2"]
        assert list(eval_manager.datasets) == ["dataset1", "dataset2"]
        assert set(eval_manager.metrics) == {"metric1", "metric2"}

    def test_get_df_for_model(self, eval_manager):
        df = eval_manager.get_df_for_model("model1")
        assert df.index.tolist() == ["model1"]

        with pytest.raises(ValueError):
            eval_manager.get_df_for_model("nonexistent_model")

    def test_get_df_for_dataset(self, eval_manager):
        df = eval_manager.get_df_for_dataset("dataset1")
        assert set(df.columns.get_level_values(0)) == {"dataset1"}

        with pytest.raises(ValueError):
            eval_manager.get_df_for_dataset("nonexistent_dataset")

    def test_get_df_for_metric(self, eval_manager):
        df = eval_manager.get_df_for_metric("metric1")
        assert set(df.columns.get_level_values(1)) == {"metric1"}

        with pytest.raises(ValueError):
            eval_manager.get_df_for_metric("nonexistent_metric")


class TestEvalManagerOperations(TestEvalManagerBase):
    def test_sort_methods(self, eval_manager):
        sorted_by_dataset = eval_manager.sort_by_dataset()
        assert list(sorted_by_dataset.datasets) == sorted(eval_manager.datasets)

        sorted_by_metric = eval_manager.sort_by_metric()
        assert list(sorted_by_metric.metrics) == sorted(eval_manager.metrics)

    def test_melt(self, eval_manager):
        melted = eval_manager.melted
        assert set(melted.columns) == {"dataset", "metric", "model", "score"}
        assert len(melted) == 6

    def test_to_csv(self, eval_manager, tmp_path):
        csv_path = tmp_path / "test_output.csv"
        eval_manager.to_csv(csv_path)
        assert csv_path.exists()

        loaded_manager = EvalManager.from_csv(csv_path)
        assert_frame_equal(loaded_manager.data, eval_manager.data)
