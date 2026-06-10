"""Tests for the RAS generation pipeline."""

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.language_models.fake import FakeListLLM
from omegaconf import OmegaConf

from autorag_research.pipelines.generation.ras import (
    RASGenerationPipeline,
    RASGenerationPipelineConfig,
    RASTriple,
    parse_ras_plan_action,
    parse_ras_triples,
)
from tests.autorag_research.pipelines.pipeline_test_utils import create_mock_retrieval_pipeline


class TrackingAsyncCallable:
    """Async callable that records a shared call order."""

    def __init__(self, label: str, calls: list[str], side_effect: list[Any] | None = None):
        self.label = label
        self.calls = calls
        self.side_effect = list(side_effect or [])
        self.await_count = 0
        self.await_args_list: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(self.label)
        self.await_count += 1
        self.await_args_list.append((args, kwargs))
        if self.side_effect:
            return self.side_effect.pop(0)
        return None


def _mock_response(content: str) -> MagicMock:
    response = MagicMock()
    response.content = content
    response.usage_metadata = {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}
    return response


def _build_unit_pipeline(llm: Any, retrieval_pipeline: Any, service: Any, **kwargs: Any) -> RASGenerationPipeline:
    with patch("autorag_research.pipelines.generation.base.BaseGenerationPipeline.__init__", return_value=None):
        pipeline = RASGenerationPipeline(
            session_factory=MagicMock(),
            name="ras_unit",
            llm=llm,
            retrieval_pipeline=retrieval_pipeline,
            **kwargs,
        )
    pipeline._llm = llm
    pipeline._retrieval_pipeline = retrieval_pipeline
    pipeline._service = service
    pipeline.pipeline_id = 1
    return pipeline


def _metadata(result: Any) -> dict[str, Any]:
    assert result.metadata is not None
    return result.metadata


class TestRASParsing:
    """Tests for RAS parsing helpers."""

    def test_parse_subquery_action(self):
        action = parse_ras_plan_action("Reasoning first. [SUBQ] apollo 11 crew")

        assert action.kind == "retrieve"
        assert action.text == "apollo 11 crew"

    def test_parse_sufficient_action(self):
        action = parse_ras_plan_action("graph is enough [sufficient]")

        assert action.kind == "sufficient"
        assert action.text == ""

    def test_parse_no_retrieval_action(self):
        action = parse_ras_plan_action("[no_retrieval]")

        assert action.kind == "no_retrieval"
        assert action.text == ""

    def test_parse_malformed_action_is_invalid(self):
        action = parse_ras_plan_action("I need Apollo 11 context but forgot the token.")

        assert action.kind == "invalid"
        assert action.text == "I need Apollo 11 context but forgot the token."

    def test_parse_xml_protocol_is_invalid(self):
        action = parse_ras_plan_action("<subquery>apollo 11 crew</subquery>")

        assert action.kind == "invalid"
        assert action.text == "<subquery>apollo 11 crew</subquery>"

    def test_parse_empty_subq_is_invalid(self):
        assert parse_ras_plan_action("[SUBQ] ").kind == "invalid"

    def test_parse_triples(self):
        triples = parse_ras_triples(
            "<triple>Apollo 11 | commander | Neil Armstrong</triple>\n<triple>Apollo 11 | landed on | Moon</triple>"
        )

        assert triples == [
            RASTriple("Apollo 11", "commander", "Neil Armstrong"),
            RASTriple("Apollo 11", "landed on", "Moon"),
        ]


class TestRASGenerationPipelineConfig:
    """Tests for RASGenerationPipelineConfig."""

    def test_ras_yaml_references_existing_llm_config(self):
        config_path = Path("configs/pipelines/generation/ras.yaml")
        config = OmegaConf.load(config_path)
        llm_name = str(config.llm)

        assert (Path("configs/llm") / f"{llm_name}.yaml").exists() or (Path("configs/llm") / f"{llm_name}.yml").exists()

    def test_get_pipeline_class(self):
        config = RASGenerationPipelineConfig(
            name="ras",
            llm=FakeListLLM(responses=["answer"]),
            retrieval_pipeline_name="bm25",
        )

        assert config.get_pipeline_class() == RASGenerationPipeline

    def test_get_pipeline_kwargs_requires_injected_retrieval_pipeline(self):
        config = RASGenerationPipelineConfig(
            name="ras",
            llm=FakeListLLM(responses=["answer"]),
            retrieval_pipeline_name="bm25",
        )

        with pytest.raises(ValueError, match="not injected"):
            config.get_pipeline_kwargs()

    def test_get_pipeline_kwargs_after_injection(self):
        retrieval_pipeline = create_mock_retrieval_pipeline(pipeline_id=44)
        llm = FakeListLLM(responses=["answer"])
        config = RASGenerationPipelineConfig(
            name="ras",
            llm=llm,
            retrieval_pipeline_name="bm25",
            max_steps=2,
            k_per_step=3,
        )

        config.inject_retrieval_pipeline(retrieval_pipeline)
        kwargs = config.get_pipeline_kwargs()

        assert kwargs["llm"] is llm
        assert kwargs["retrieval_pipeline"] is retrieval_pipeline
        assert kwargs["max_steps"] == 2
        assert kwargs["k_per_step"] == 3
        assert "no_retrieval_prompt_template" in kwargs


class TestRASGenerationPipeline:
    """Tests for RAS pipeline behavior."""

    def test_initialization_rejects_invalid_steps(self):
        with (
            patch("autorag_research.pipelines.generation.base.BaseGenerationPipeline.__init__", return_value=None),
            pytest.raises(ValueError, match="max_steps must be >= 1"),
        ):
            RASGenerationPipeline(
                session_factory=MagicMock(),
                name="invalid_ras",
                llm=FakeListLLM(responses=["answer"]),
                retrieval_pipeline=create_mock_retrieval_pipeline(),
                max_steps=0,
            )

    @pytest.mark.asyncio
    async def test_no_retrieval_first_plan_answers_directly_without_retrieval(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(side_effect=[_mock_response("[NO_RETRIEVAL]"), _mock_response("Parametric answer.")])
        retrieval_pipeline = create_mock_retrieval_pipeline()
        retrieval_pipeline._retrieve_by_id = AsyncMock(return_value=[])
        retrieval_pipeline.retrieve = AsyncMock(return_value=[])
        service = MagicMock()
        service.get_query_text.return_value = "What is Python?"
        pipeline = _build_unit_pipeline(llm, retrieval_pipeline, service)

        result = await pipeline._generate(1, top_k=2)
        metadata = _metadata(result)

        retrieval_pipeline._retrieve_by_id.assert_not_awaited()
        retrieval_pipeline.retrieve.assert_not_awaited()
        assert result.text == "Parametric answer."
        assert metadata["route"] == "no_retrieval"
        assert metadata["retrieved_chunk_ids"] == []
        assert llm.ainvoke.await_args_list[-1].args[0].count("What is Python?") == 1

    @pytest.mark.asyncio
    async def test_plan_first_ordering_retrieves_only_after_first_planning_call(self):
        call_order: list[str] = []
        llm = MagicMock()
        llm.ainvoke = TrackingAsyncCallable(
            "llm",
            call_order,
            [_mock_response("[SUBQ] apollo 11 crew"), _mock_response("[SUFFICIENT]"), _mock_response("answer")],
        )
        retrieval_pipeline = create_mock_retrieval_pipeline()
        retrieval_pipeline._retrieve_by_id = TrackingAsyncCallable("retrieve_by_id", call_order, [[]])
        retrieval_pipeline.retrieve = TrackingAsyncCallable("retrieve", call_order, [[]])
        service = MagicMock()
        service.get_query_text.return_value = "Who commanded Apollo 11?"
        pipeline = _build_unit_pipeline(llm, retrieval_pipeline, service, max_steps=2)

        await pipeline._generate(1, top_k=2)

        assert call_order[:2] == ["llm", "retrieve"]
        assert "retrieve_by_id" not in call_order

    @pytest.mark.asyncio
    async def test_subquery_loop_extracts_history_and_answers_without_raw_passages(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            side_effect=[
                _mock_response("[SUBQ] q1"),
                _mock_response("<triple>Apollo 11 | commander | Neil Armstrong</triple>"),
                _mock_response("[SUFFICIENT]"),
                _mock_response("Neil Armstrong commanded Apollo 11."),
            ]
        )
        retrieval_pipeline = create_mock_retrieval_pipeline()
        retrieval_pipeline._retrieve_by_id = AsyncMock(return_value=[])
        retrieval_pipeline.retrieve = AsyncMock(
            return_value=[{"doc_id": 10, "score": 0.9, "content": "RAW PASSAGE: Neil Armstrong commanded Apollo 11."}]
        )
        service = MagicMock()
        service.get_query_text.return_value = "Who commanded Apollo 11?"
        pipeline = _build_unit_pipeline(llm, retrieval_pipeline, service, max_steps=3, k_per_step=2)

        result = await pipeline._generate(1, top_k=2)
        metadata = _metadata(result)
        answer_prompt = llm.ainvoke.await_args_list[-1].args[0]

        retrieval_pipeline._retrieve_by_id.assert_not_awaited()
        retrieval_pipeline.retrieve.assert_awaited_once_with("q1", 2)
        assert result.text == "Neil Armstrong commanded Apollo 11."
        assert metadata["route"] == "graph"
        assert metadata["subqueries"] == ["q1"]
        assert metadata["retrieved_chunk_ids"] == [10]
        assert metadata["triples"] == [("Apollo 11", "commander", "Neil Armstrong")]
        assert metadata["iteration_history"] == [
            {"subquery": "q1", "triples": [("Apollo 11", "commander", "Neil Armstrong")]}
        ]
        assert "Structured graph G_Q" in answer_prompt
        assert "Subquery-to-triples history" in answer_prompt
        assert "Neil Armstrong" in answer_prompt
        assert "RAW PASSAGE" not in answer_prompt

    @pytest.mark.asyncio
    async def test_no_new_passages_continues_to_next_planning_call(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            side_effect=[
                _mock_response("[SUBQ] q1"),
                _mock_response("<triple>A | relates to | B</triple>"),
                _mock_response("[SUBQ] q1 again"),
                _mock_response("[SUFFICIENT]"),
                _mock_response("answer"),
            ]
        )
        retrieval_pipeline = create_mock_retrieval_pipeline()
        retrieval_pipeline._retrieve_by_id = AsyncMock(return_value=[])
        retrieval_pipeline.retrieve = AsyncMock(
            side_effect=[
                [{"doc_id": 5, "score": 0.9, "content": "A relates to B."}],
                [{"doc_id": 5, "score": 0.9, "content": "A relates to B."}],
            ]
        )
        service = MagicMock()
        service.get_query_text.return_value = "Question?"
        pipeline = _build_unit_pipeline(llm, retrieval_pipeline, service, max_steps=3, k_per_step=1)

        result = await pipeline._generate(1, top_k=1)
        metadata = _metadata(result)

        assert llm.ainvoke.await_count == 5
        assert retrieval_pipeline.retrieve.await_count == 2
        assert metadata["iteration_history"][-1] == {"subquery": "q1 again", "triples": [], "note": "no new passages"}
        assert metadata["plan_trace"][-2] == "note: no new passages for q1 again"

    @pytest.mark.asyncio
    async def test_generate_backfills_missing_content(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            side_effect=[
                _mock_response("[SUBQ] missing content"),
                _mock_response("<triple>A | relates to | B</triple>"),
                _mock_response("answer"),
            ]
        )
        retrieval_pipeline = create_mock_retrieval_pipeline()
        retrieval_pipeline._retrieve_by_id = AsyncMock(return_value=[])
        retrieval_pipeline.retrieve = AsyncMock(return_value=[{"doc_id": 5, "score": 0.8}])
        service = MagicMock()
        service.get_query_text.return_value = "Question?"
        service.get_chunk_contents.return_value = ["Fetched passage."]
        pipeline = _build_unit_pipeline(llm, retrieval_pipeline, service, max_steps=1, k_per_step=1)

        result = await pipeline._generate(1, top_k=1)
        metadata = _metadata(result)

        service.get_chunk_contents.assert_called_once_with([5])
        assert metadata["evidence"] == ["Fetched passage."]
        assert metadata["triples"] == [("A", "relates to", "B")]

    @pytest.mark.asyncio
    async def test_malformed_plan_retries_once_then_sufficiency_without_original_query_retrieval(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            side_effect=[
                _mock_response("Need more evidence but omitted required token."),
                _mock_response("Still malformed."),
                _mock_response("Answer from current graph."),
            ]
        )
        retrieval_pipeline = create_mock_retrieval_pipeline()
        retrieval_pipeline._retrieve_by_id = AsyncMock(return_value=[])
        retrieval_pipeline.retrieve = AsyncMock(return_value=[])
        service = MagicMock()
        service.get_query_text.return_value = "Who commanded Apollo 11?"
        pipeline = _build_unit_pipeline(llm, retrieval_pipeline, service, max_steps=1, k_per_step=2)

        result = await pipeline._generate(1, top_k=2)
        metadata = _metadata(result)

        retrieval_pipeline._retrieve_by_id.assert_not_awaited()
        retrieval_pipeline.retrieve.assert_not_awaited()
        assert result.text == "Answer from current graph."
        assert metadata["subqueries"] == []
        assert metadata["malformed_plans"] == ["Need more evidence but omitted required token.", "Still malformed."]
        assert metadata["plan_trace"] == [
            "invalid: Need more evidence but omitted required token.",
            "retry invalid: Still malformed.",
        ]

    def test_merge_triples_deduplicates(self):
        pipeline = _build_unit_pipeline(MagicMock(), create_mock_retrieval_pipeline(), MagicMock())
        triple = RASTriple("A", "p", "B")

        assert pipeline._merge_triples([triple], [triple]) == [triple]
