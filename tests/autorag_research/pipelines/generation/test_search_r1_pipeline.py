"""Tests for the inference-only Search-R1 generation pipeline."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hydra.utils import instantiate
from langchain_core.language_models.fake import FakeListLLM
from omegaconf import OmegaConf

from autorag_research.pipelines.generation.search_r1 import (
    SearchR1GenerationPipeline,
    SearchR1GenerationPipelineConfig,
    parse_search_r1_action,
)
from tests.autorag_research.pipelines.pipeline_test_utils import create_mock_retrieval_pipeline

RETHINK_MESSAGE = "My action is not correct. Let me rethink."


def _mock_response(content: str) -> MagicMock:
    response = MagicMock()
    response.content = content
    response.usage_metadata = {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}
    return response


def _build_unit_pipeline(
    llm: Any,
    retrieval_pipeline: Any,
    service: Any,
    **kwargs: Any,
) -> SearchR1GenerationPipeline:
    with patch("autorag_research.pipelines.generation.base.BaseGenerationPipeline.__init__", return_value=None):
        pipeline = SearchR1GenerationPipeline(
            session_factory=MagicMock(),
            name="search_r1_unit",
            llm=llm,
            retrieval_pipeline=retrieval_pipeline,
            **kwargs,
        )
    pipeline._llm = llm
    pipeline._retrieval_pipeline = retrieval_pipeline
    pipeline._service = service
    pipeline.pipeline_id = 1
    return pipeline


class TestSearchR1ActionParsing:
    """Tests for Search-R1 action parsing."""

    def test_parse_search_action(self):
        action = parse_search_r1_action("Need evidence. <search>latest mars mission</search>")

        assert action.kind == "search"
        assert action.text == "latest mars mission"

    def test_parse_answer_action(self):
        action = parse_search_r1_action("<answer>Mars is the fourth planet.</answer>")

        assert action.kind == "answer"
        assert action.text == "Mars is the fourth planet."

    def test_parse_earliest_completed_action(self):
        action = parse_search_r1_action("<search>q</search> blah <answer>x</answer>")

        assert action.kind == "search"
        assert action.text == "q"
        assert action.end_index == len("<search>q</search>")

    def test_parse_plain_response_as_malformed(self):
        action = parse_search_r1_action("Plain final answer")

        assert action.kind == "malformed"
        assert action.text == ""
        assert action.end_index is None


class TestSearchR1GenerationPipelineConfig:
    """Tests for SearchR1GenerationPipelineConfig."""

    def test_get_pipeline_class(self):
        config = SearchR1GenerationPipelineConfig(
            name="search_r1",
            llm=FakeListLLM(responses=["<answer>ok</answer>"]),
            retrieval_pipeline_name="bm25",
        )

        assert config.get_pipeline_class() == SearchR1GenerationPipeline

    def test_get_pipeline_kwargs_requires_injected_retrieval_pipeline(self):
        config = SearchR1GenerationPipelineConfig(
            name="search_r1",
            llm=FakeListLLM(responses=["<answer>ok</answer>"]),
            retrieval_pipeline_name="bm25",
        )

        with pytest.raises(ValueError, match="not injected"):
            config.get_pipeline_kwargs()

    def test_get_pipeline_kwargs_after_injection(self):
        wrapped_retrieval = create_mock_retrieval_pipeline(pipeline_id=77)
        llm = FakeListLLM(responses=["<answer>ok</answer>"])
        config = SearchR1GenerationPipelineConfig(
            name="search_r1",
            llm=llm,
            retrieval_pipeline_name="bm25",
            max_actions=4,
            k_per_search=3,
        )

        config.inject_retrieval_pipeline(wrapped_retrieval)
        kwargs = config.get_pipeline_kwargs()

        assert kwargs["llm"] is llm
        assert kwargs["retrieval_pipeline"] is wrapped_retrieval
        assert kwargs["max_actions"] == 4
        assert kwargs["k_per_search"] == 3
        assert kwargs["fallback_to_final_prompt"] is False

    def test_shipped_yaml_uses_paper_defaults(self):
        with patch("autorag_research.injection.load_llm", return_value=FakeListLLM(responses=["<answer>ok</answer>"])):
            config = instantiate(OmegaConf.load("configs/pipelines/generation/search_r1.yaml"))

        assert isinstance(config, SearchR1GenerationPipelineConfig)
        assert config.max_actions == 4
        assert config.k_per_search == 3
        assert config.fallback_to_final_prompt is False


class TestSearchR1GenerationPipeline:
    """Tests for SearchR1GenerationPipeline behavior."""

    def test_initialization_rejects_invalid_action_budget(self):
        with (
            patch("autorag_research.pipelines.generation.base.BaseGenerationPipeline.__init__", return_value=None),
            pytest.raises(ValueError, match="max_actions must be >= 1"),
        ):
            SearchR1GenerationPipeline(
                session_factory=MagicMock(),
                name="invalid_search_r1",
                llm=FakeListLLM(responses=["<answer>ok</answer>"]),
                retrieval_pipeline=create_mock_retrieval_pipeline(),
                max_actions=0,
            )

    def test_prompt_template_requires_query_placeholder(self):
        with (
            patch("autorag_research.pipelines.generation.base.BaseGenerationPipeline.__init__", return_value=None),
            pytest.raises(ValueError, match="step_prompt_template must contain"),
        ):
            SearchR1GenerationPipeline(
                session_factory=MagicMock(),
                name="invalid_prompt_search_r1",
                llm=FakeListLLM(responses=["<answer>ok</answer>"]),
                retrieval_pipeline=create_mock_retrieval_pipeline(),
                step_prompt_template="No placeholder",
            )

    @pytest.mark.asyncio
    async def test_rollout_searches_then_answers(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            side_effect=[
                _mock_response("<think>t</think><search>q1</search>"),
                _mock_response("<think>t2</think><answer>final</answer>"),
            ]
        )
        retrieval_pipeline = create_mock_retrieval_pipeline(
            default_results=[{"doc_id": 10, "score": 0.9, "content": "Mars has Phobos and Deimos."}]
        )
        service = MagicMock()
        service.get_query_text.return_value = "How many moons does Mars have?"
        pipeline = _build_unit_pipeline(llm, retrieval_pipeline, service, max_actions=4, k_per_search=3)

        result = await pipeline._generate(1, top_k=5)

        retrieval_pipeline.retrieve.assert_awaited_once_with("q1", 3)
        second_prompt = llm.ainvoke.await_args_list[1].args[0]
        assert "<think>t</think><search>q1</search>" in second_prompt
        assert "<information>Mars has Phobos and Deimos.</information>" in second_prompt
        assert result.text == "final"
        assert result.metadata["rollout"] == (
            "<think>t</think><search>q1</search>"
            "<information>Mars has Phobos and Deimos.</information>"
            "<think>t2</think><answer>final</answer>"
        )
        assert result.metadata["search_queries"] == ["q1"]
        assert result.metadata["retrieved_chunk_ids"] == [10]
        assert result.metadata["actions_used"] == 2
        assert result.metadata["terminated_by"] == "answer"
        assert result.token_usage == {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10}

    @pytest.mark.asyncio
    async def test_earliest_action_dispatches_search_and_discards_trailing_text(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            side_effect=[
                _mock_response("<search>q</search> blah <answer>x</answer>"),
                _mock_response("<answer>final</answer>"),
            ]
        )
        retrieval_pipeline = create_mock_retrieval_pipeline(default_results=[{"doc_id": 1, "content": "Evidence."}])
        service = MagicMock()
        service.get_query_text.return_value = "Question?"
        pipeline = _build_unit_pipeline(llm, retrieval_pipeline, service, max_actions=2)

        result = await pipeline._generate(1, top_k=3)

        retrieval_pipeline.retrieve.assert_awaited_once_with("q", 3)
        assert "blah" not in result.metadata["rollout"]
        assert "<answer>x</answer>" not in result.metadata["rollout"]
        assert result.metadata["rollout"].startswith("<search>q</search><information>Evidence.</information>")
        assert result.text == "final"

    @pytest.mark.asyncio
    async def test_malformed_segment_appends_rethink_and_counts_budget(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            side_effect=[
                _mock_response("untagged segment"),
                _mock_response("<answer>corrected</answer>"),
            ]
        )
        retrieval_pipeline = create_mock_retrieval_pipeline(default_results=[])
        service = MagicMock()
        service.get_query_text.return_value = "Question?"
        pipeline = _build_unit_pipeline(llm, retrieval_pipeline, service, max_actions=2)

        result = await pipeline._generate(1, top_k=3)

        second_prompt = llm.ainvoke.await_args_list[1].args[0]
        assert "untagged segment" in second_prompt
        assert RETHINK_MESSAGE in second_prompt
        assert result.text == "corrected"
        assert result.metadata["actions_used"] == 2
        retrieval_pipeline.retrieve.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_budget_exhaustion_without_fallback_returns_empty_answer(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=_mock_response("<search>first query</search>"))
        retrieval_pipeline = create_mock_retrieval_pipeline(
            default_results=[{"doc_id": 1, "score": 0.8, "content": "Evidence."}]
        )
        service = MagicMock()
        service.get_query_text.return_value = "Question?"
        pipeline = _build_unit_pipeline(llm, retrieval_pipeline, service, max_actions=1, k_per_search=3)

        result = await pipeline._generate(1, top_k=2)

        retrieval_pipeline.retrieve.assert_awaited_once_with("first query", 2)
        assert result.text == ""
        assert result.metadata["terminated_by"] == "budget_exhausted"
        assert result.metadata["rollout"] == "<search>first query</search><information>Evidence.</information>"
        assert llm.ainvoke.await_count == 1

    @pytest.mark.asyncio
    async def test_budget_exhaustion_with_fallback_makes_final_prompt_call(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            side_effect=[
                _mock_response("<search>first query</search>"),
                _mock_response("Fallback final answer."),
            ]
        )
        retrieval_pipeline = create_mock_retrieval_pipeline(
            default_results=[{"doc_id": 1, "score": 0.8, "content": "Evidence."}]
        )
        service = MagicMock()
        service.get_query_text.return_value = "Question?"
        pipeline = _build_unit_pipeline(
            llm, retrieval_pipeline, service, max_actions=1, k_per_search=3, fallback_to_final_prompt=True
        )

        result = await pipeline._generate(1, top_k=2)

        assert result.text == "Fallback final answer."
        assert result.metadata["terminated_by"] == "budget_exhausted_fallback"
        assert "Rollout:" in llm.ainvoke.await_args_list[1].args[0]
        assert llm.ainvoke.await_count == 2

    @pytest.mark.asyncio
    async def test_generate_backfills_missing_result_contents(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            side_effect=[
                _mock_response("<search>missing content query</search>"),
                _mock_response("<answer>Done.</answer>"),
            ]
        )
        retrieval_pipeline = create_mock_retrieval_pipeline(default_results=[{"doc_id": 5, "score": 0.8}])
        service = MagicMock()
        service.get_query_text.return_value = "Question?"
        service.get_chunk_contents.return_value = ["Fetched evidence."]
        pipeline = _build_unit_pipeline(llm, retrieval_pipeline, service, max_actions=2, k_per_search=1)

        result = await pipeline._generate(1, top_k=1)

        service.get_chunk_contents.assert_called_once_with([5])
        assert "<information>Fetched evidence.</information>" in result.metadata["rollout"]
        assert result.metadata["retrieved_chunk_ids"] == [5]
