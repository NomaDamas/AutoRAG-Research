"""Tests for Hybrid Deep Searcher generation pipeline."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.language_models.fake import FakeListLLM

from autorag_research.pipelines.generation.hybrid_deep_searcher import (
    HybridDeepSearcherPipeline,
    HybridDeepSearcherPipelineConfig,
    parse_hybrid_deep_search_action,
)
from tests.autorag_research.pipelines.pipeline_test_utils import create_mock_retrieval_pipeline


def _mock_response(content: str) -> MagicMock:
    response = MagicMock()
    response.content = content
    response.usage_metadata = {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}
    return response


def _build_unit_pipeline(llm: Any, retrieval_pipeline: Any, service: Any, **kwargs: Any) -> HybridDeepSearcherPipeline:
    with patch("autorag_research.pipelines.generation.base.BaseGenerationPipeline.__init__", return_value=None):
        pipeline = HybridDeepSearcherPipeline(
            session_factory=MagicMock(),
            name="hds_unit",
            llm=llm,
            retrieval_pipeline=retrieval_pipeline,
            **kwargs,
        )
    pipeline._llm = llm
    pipeline._retrieval_pipeline = retrieval_pipeline
    pipeline._service = service
    pipeline.pipeline_id = 1
    return pipeline


class TestHybridDeepSearchActionParsing:
    """Tests for action parsing."""

    def test_parse_parallel_queries(self):
        action = parse_hybrid_deep_search_action("<queries>\n1. alpha\n- beta\n</queries>", max_queries=4)

        assert action.kind == "queries"
        assert action.queries == ("alpha", "beta")

    def test_parse_limits_queries(self):
        action = parse_hybrid_deep_search_action("<queries>\na\nb\nc\n</queries>", max_queries=2)

        assert action.queries == ("a", "b")

    def test_parse_answer(self):
        action = parse_hybrid_deep_search_action("<answer>done</answer>", max_queries=2)

        assert action.kind == "answer"
        assert action.text == "done"


class TestHybridDeepSearcherPipelineConfig:
    """Tests for HybridDeepSearcherPipelineConfig."""

    def test_get_pipeline_class(self):
        config = HybridDeepSearcherPipelineConfig(
            name="hybrid_deep_searcher",
            llm=FakeListLLM(responses=["<answer>ok</answer>"]),
            retrieval_pipeline_name="hybrid_rrf",
        )

        assert config.get_pipeline_class() == HybridDeepSearcherPipeline

    def test_get_pipeline_kwargs_requires_injected_retrieval_pipeline(self):
        config = HybridDeepSearcherPipelineConfig(
            name="hybrid_deep_searcher",
            llm=FakeListLLM(responses=["<answer>ok</answer>"]),
            retrieval_pipeline_name="hybrid_rrf",
        )

        with pytest.raises(ValueError, match="not injected"):
            config.get_pipeline_kwargs()

    def test_get_pipeline_kwargs_after_injection(self):
        retrieval_pipeline = create_mock_retrieval_pipeline(pipeline_id=88)
        llm = FakeListLLM(responses=["<answer>ok</answer>"])
        config = HybridDeepSearcherPipelineConfig(
            name="hybrid_deep_searcher",
            llm=llm,
            retrieval_pipeline_name="hybrid_rrf",
            max_turns=4,
            max_parallel_queries=3,
        )

        config.inject_retrieval_pipeline(retrieval_pipeline)
        kwargs = config.get_pipeline_kwargs()

        assert kwargs["llm"] is llm
        assert kwargs["retrieval_pipeline"] is retrieval_pipeline
        assert kwargs["max_turns"] == 4
        assert kwargs["max_parallel_queries"] == 3


class TestHybridDeepSearcherPipeline:
    """Tests for HDS generation behavior."""

    def test_initialization_rejects_invalid_turns(self):
        with (
            patch("autorag_research.pipelines.generation.base.BaseGenerationPipeline.__init__", return_value=None),
            pytest.raises(ValueError, match="max_turns must be >= 1"),
        ):
            HybridDeepSearcherPipeline(
                session_factory=MagicMock(),
                name="invalid_hds",
                llm=FakeListLLM(responses=["<answer>ok</answer>"]),
                retrieval_pipeline=create_mock_retrieval_pipeline(),
                max_turns=0,
            )

    @pytest.mark.asyncio
    async def test_generate_parallel_search_then_answer(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            side_effect=[
                _mock_response("<queries>\nmoon origin\nmoon composition\n</queries>"),
                _mock_response("<answer>The Moon likely formed from giant-impact debris.</answer>"),
            ]
        )
        retrieval_pipeline = create_mock_retrieval_pipeline()
        retrieval_pipeline.retrieve = AsyncMock(
            side_effect=[
                [{"doc_id": 1, "score": 0.9, "content": "Giant impact evidence."}],
                [{"doc_id": 2, "score": 0.8, "content": "Moon composition evidence."}],
            ]
        )
        service = MagicMock()
        service.get_query_text.return_value = "How did the Moon form?"
        pipeline = _build_unit_pipeline(llm, retrieval_pipeline, service, max_turns=3, k_per_query=4)

        result = await pipeline._generate(1, top_k=2)

        retrieval_pipeline.retrieve.assert_any_await("moon origin", 2)
        retrieval_pipeline.retrieve.assert_any_await("moon composition", 2)
        assert result.text == "The Moon likely formed from giant-impact debris."
        assert result.metadata["retrieved_chunk_ids"] == [1, 2]
        assert result.metadata["terminated_by"] == "answer"
        assert result.token_usage == {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10}

    @pytest.mark.asyncio
    async def test_generate_merges_duplicate_results_and_falls_back(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            side_effect=[
                _mock_response("<queries>\na\nb\n</queries>"),
                _mock_response("fallback answer"),
            ]
        )
        retrieval_pipeline = create_mock_retrieval_pipeline()
        retrieval_pipeline.retrieve = AsyncMock(
            side_effect=[
                [{"doc_id": 1, "score": 0.2, "content": "low"}],
                [{"doc_id": 1, "score": 0.9, "content": "high"}],
            ]
        )
        service = MagicMock()
        service.get_query_text.return_value = "Question?"
        pipeline = _build_unit_pipeline(llm, retrieval_pipeline, service, max_turns=1, k_per_query=3)

        result = await pipeline._generate(1, top_k=3)

        assert result.text == "fallback answer"
        assert result.metadata["retrieved_chunk_ids"] == [1]
        assert result.metadata["evidence"] == ["high"]
        assert result.metadata["terminated_by"] == "max_turns_fallback"

    @pytest.mark.asyncio
    async def test_generate_backfills_missing_contents(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            side_effect=[
                _mock_response("<queries>\na\n</queries>"),
                _mock_response("<answer>done</answer>"),
            ]
        )
        retrieval_pipeline = create_mock_retrieval_pipeline(default_results=[{"doc_id": 7, "score": 0.9}])
        service = MagicMock()
        service.get_query_text.return_value = "Question?"
        service.get_chunk_contents.return_value = ["Fetched content."]
        pipeline = _build_unit_pipeline(llm, retrieval_pipeline, service, max_turns=2, k_per_query=1)

        result = await pipeline._generate(1, top_k=1)

        service.get_chunk_contents.assert_called_once_with([7])
        assert result.metadata["evidence"] == ["Fetched content."]
