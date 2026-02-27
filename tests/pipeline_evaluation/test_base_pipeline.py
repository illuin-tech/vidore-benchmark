"""
Tests for base pipeline definition.
"""

from typing import Dict, List

import pytest

from vidore_benchmark.pipeline_evaluation.base_pipeline import BasePipeline


class TestBasePipeline:
    def test_base_pipeline_is_abstract(self):
        """Test that BasePipeline cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            BasePipeline()

        assert "abstract" in str(exc_info.value).lower()

    def test_base_pipeline_requires_search_method(self):
        """Test that subclasses must implement search method."""

        class IncompletePipeline(BasePipeline):
            pass

        with pytest.raises(TypeError) as exc_info:
            IncompletePipeline()

        assert "search" in str(exc_info.value)

    def test_complete_subclass_can_be_instantiated(self):
        """Test that a properly implemented subclass can be instantiated."""

        class CompletePipeline(BasePipeline):
            def search(
                self,
                query_ids: List[str],
                queries: List[str],
            ) -> Dict[str, Dict[str, float]]:
                return {}

        pipeline = CompletePipeline()
        assert isinstance(pipeline, BasePipeline)

    def test_search_method_signature(self):
        """Test that search method accepts correct parameters."""

        class TestPipeline(BasePipeline):
            def search(
                self,
                query_ids: List[str],
                queries: List[str],
            ) -> Dict[str, Dict[str, float]]:
                # Return scores for each query
                return {qid: {"doc1": 0.5} for qid in query_ids}

        pipeline = TestPipeline()

        query_ids = ["q1", "q2"]
        queries = ["Query 1", "Query 2"]

        results = pipeline.search(query_ids, queries)

        assert isinstance(results, dict)
        assert set(results.keys()) == {"q1", "q2"}
        assert set(results["q1"].keys()) == {"doc1"}

    def test_search_can_return_tuple_with_infos(self):
        """Test that search method can return tuple with optional infos."""

        class TestPipelineWithInfos(BasePipeline):
            def search(
                self,
                query_ids: List[str],
                queries: List[str],
            ):
                results = {qid: {"doc1": 0.9} for qid in query_ids}
                infos = {"cost": 0.50, "num_gpus": 1}
                return results, infos

        pipeline = TestPipelineWithInfos()

        result = pipeline.search(["q1"], ["Query"])

        assert isinstance(result, tuple)
        assert len(result) == 2
        results, infos = result
        assert isinstance(results, dict)
        assert isinstance(infos, dict)
        assert infos["cost"] == 0.50

    def test_search_with_empty_inputs(self):
        """Test that search handles empty inputs gracefully."""

        class TestPipeline(BasePipeline):
            def search(
                self,
                query_ids: List[str],
                queries: List[str],
            ) -> Dict[str, Dict[str, float]]:
                return {qid: {} for qid in query_ids}

        pipeline = TestPipeline()

        # Empty corpus
        pipeline.index([], [], [])
        results = pipeline.search(["q1"], ["Query"])
        assert results == {"q1": {}}

        # Empty queries
        pipeline.index(["doc1"], [None], ["Text"])
        results = pipeline.search([], [])
        assert results == {}

    def test_search_partial_results(self):
        """Test that search can return results for only some corpus items."""

        class PartialResultsPipeline(BasePipeline):
            def search(
                self,
                query_ids: List[str],
                queries: List[str],
            ) -> Dict[str, Dict[str, float]]:
                # Only return top-k results (first 2 corpus items)
                return {qid: {cid: 0.5 for cid in self.corpus_ids[:2]} for qid in query_ids}

        pipeline = PartialResultsPipeline()

        pipeline.index(
            ["doc1", "doc2", "doc3", "doc4", "doc5"],
            [None] * 5,
            ["Text"] * 5,
        )

        results = pipeline.search(
            ["q1"],
            ["Query"],
        )

        # Should only have 2 results per query
        assert len(results["q1"]) == 2
        assert "doc1" in results["q1"]
        assert "doc2" in results["q1"]
        assert "doc3" not in results["q1"]

    def test_index_method_stores_corpus(self):
        """Test that default index method stores corpus data."""

        class DefaultIndexPipeline(BasePipeline):
            def search(self, query_ids, queries):
                return {}

        pipeline = DefaultIndexPipeline()

        corpus_ids = ["doc1", "doc2"]
        corpus_images = [None, None]
        corpus_texts = ["Text 1", "Text 2"]

        pipeline.index(corpus_ids, corpus_images, corpus_texts)

        assert pipeline.corpus_ids == corpus_ids
        assert pipeline.corpus_images == corpus_images
        assert pipeline.corpus_texts == corpus_texts
