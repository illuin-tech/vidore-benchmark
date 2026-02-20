"""
Tests for base pipeline definition.
"""

from typing import Any, Dict, List

import pytest

from vidore_benchmark.pipeline_evaluation.base_pipeline import BasePipeline


class TestBasePipeline:
    def test_base_pipeline_is_abstract(self):
        """Test that BasePipeline cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            BasePipeline()

        assert "abstract" in str(exc_info.value).lower()

    def test_base_pipeline_requires_retrieve_method(self):
        """Test that subclasses must implement retrieve method."""

        class IncompletePipeline(BasePipeline):
            pass

        with pytest.raises(TypeError) as exc_info:
            IncompletePipeline()

        assert "retrieve" in str(exc_info.value)

    def test_complete_subclass_can_be_instantiated(self):
        """Test that a properly implemented subclass can be instantiated."""

        class CompletePipeline(BasePipeline):
            def retrieve(
                self,
                query_ids: List[str],
                queries: List[str],
                corpus_ids: List[str],
                corpus_images: List[Any],
                corpus_texts: List[str],
            ) -> Dict[str, Dict[str, float]]:
                return {}

        pipeline = CompletePipeline()
        assert isinstance(pipeline, BasePipeline)

    def test_retrieve_method_signature(self):
        """Test that retrieve method accepts correct parameters."""

        class TestPipeline(BasePipeline):
            def retrieve(
                self,
                query_ids: List[str],
                queries: List[str],
                corpus_ids: List[str],
                corpus_images: List[Any],
                corpus_texts: List[str],
            ) -> Dict[str, Dict[str, float]]:
                # Return scores for each query
                return {qid: {cid: 0.5 for cid in corpus_ids} for qid in query_ids}

        pipeline = TestPipeline()

        query_ids = ["q1", "q2"]
        queries = ["Query 1", "Query 2"]
        corpus_ids = ["doc1", "doc2", "doc3"]
        corpus_images = [None, None, None]  # Mock images
        corpus_texts = ["Text 1", "Text 2", "Text 3"]

        results = pipeline.retrieve(query_ids, queries, corpus_ids, corpus_images, corpus_texts)

        assert isinstance(results, dict)
        assert set(results.keys()) == {"q1", "q2"}
        assert set(results["q1"].keys()) == {"doc1", "doc2", "doc3"}

    def test_retrieve_can_return_tuple_with_infos(self):
        """Test that retrieve method can return tuple with optional infos."""

        class TestPipelineWithInfos(BasePipeline):
            def retrieve(
                self,
                query_ids: List[str],
                queries: List[str],
                corpus_ids: List[str],
                corpus_images: List[Any],
                corpus_texts: List[str],
            ):
                results = {qid: {corpus_ids[0]: 0.9} for qid in query_ids}
                infos = {"cost": 0.50, "num_gpus": 1}
                return results, infos

        pipeline = TestPipelineWithInfos()

        result = pipeline.retrieve(["q1"], ["Query"], ["doc1"], [None], ["Text"])

        assert isinstance(result, tuple)
        assert len(result) == 2
        results, infos = result
        assert isinstance(results, dict)
        assert isinstance(infos, dict)
        assert infos["cost"] == 0.50

    def test_retrieve_with_empty_inputs(self):
        """Test that retrieve handles empty inputs gracefully."""

        class TestPipeline(BasePipeline):
            def retrieve(
                self,
                query_ids: List[str],
                queries: List[str],
                corpus_ids: List[str],
                corpus_images: List[Any],
                corpus_texts: List[str],
            ) -> Dict[str, Dict[str, float]]:
                return {qid: {} for qid in query_ids}

        pipeline = TestPipeline()

        # Empty corpus
        results = pipeline.retrieve(["q1"], ["Query"], [], [], [])
        assert results == {"q1": {}}

        # Empty queries
        results = pipeline.retrieve([], [], ["doc1"], [None], ["Text"])
        assert results == {}

    def test_retrieve_partial_results(self):
        """Test that retrieve can return results for only some corpus items."""

        class PartialResultsPipeline(BasePipeline):
            def retrieve(
                self,
                query_ids: List[str],
                queries: List[str],
                corpus_ids: List[str],
                corpus_images: List[Any],
                corpus_texts: List[str],
            ) -> Dict[str, Dict[str, float]]:
                # Only return top-k results (first 2 corpus items)
                return {qid: {cid: 0.5 for cid in corpus_ids[:2]} for qid in query_ids}

        pipeline = PartialResultsPipeline()

        results = pipeline.retrieve(
            ["q1"],
            ["Query"],
            ["doc1", "doc2", "doc3", "doc4", "doc5"],
            [None] * 5,
            ["Text"] * 5,
        )

        # Should only have 2 results per query
        assert len(results["q1"]) == 2
        assert "doc1" in results["q1"]
        assert "doc2" in results["q1"]
        assert "doc3" not in results["q1"]
