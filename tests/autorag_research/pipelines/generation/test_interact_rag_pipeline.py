"""Tests for the INTERACT-RAG generation pipeline."""

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.language_models.fake import FakeListLLM
from omegaconf import OmegaConf

from autorag_research.pipelines.generation.interact_rag import (
    InteractRAGPipeline,
    InteractRAGPipelineConfig,
    InteractRAGState,
    parse_doc_ids,
    parse_interact_rag_action,
    parse_interact_rag_actions,
)
from tests.autorag_research.pipelines.pipeline_test_utils import create_mock_retrieval_pipeline


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
) -> InteractRAGPipeline:
    with patch("autorag_research.pipelines.generation.base.BaseGenerationPipeline.__init__", return_value=None):
        pipeline = InteractRAGPipeline(
            session_factory=MagicMock(),
            name="interact_rag_unit",
            llm=llm,
            retrieval_pipeline=retrieval_pipeline,
            **kwargs,
        )
    pipeline._llm = llm
    pipeline._retrieval_pipeline = retrieval_pipeline
    pipeline._service = service
    pipeline.pipeline_id = 1
    if "exact_retrieval_pipeline" in kwargs:
        pipeline._exact_retrieval_pipeline = kwargs["exact_retrieval_pipeline"]
    return pipeline


class TestInteractRAGParsing:
    """Tests for action parsing helpers."""

    def test_parse_semantic_search(self):
        action = parse_interact_rag_action("<semantic_search>moon landing timeline</semantic_search>")

        assert action.kind == "semantic_search"
        assert action.text == "moon landing timeline"

    def test_parse_weighted_fusion_with_weights(self):
        action = parse_interact_rag_action('<weighted_fusion semantic="0.7" exact="0.3">apollo 11</weighted_fusion>')

        assert action.kind == "weighted_fusion"
        assert action.text == "apollo 11"
        assert action.semantic_weight == 0.7
        assert action.exact_weight == 0.3

    def test_parse_weighted_fusion_malformed_weights_falls_back_to_defaults(self):
        action = parse_interact_rag_action('<weighted_fusion semantic="." exact="0.2">apollo 11</weighted_fusion>')

        assert action.kind == "weighted_fusion"
        assert action.text == "apollo 11"
        assert action.semantic_weight is None
        assert action.exact_weight is None

    def test_parse_answer_preferred(self):
        action = parse_interact_rag_action("<semantic_search>x</semantic_search><answer>final</answer>")

        assert action.kind == "answer"
        assert action.text == "final"

    def test_parse_multiple_actions_in_order(self):
        actions = parse_interact_rag_actions(
            "reason <exclude_docs>2</exclude_docs> then <semantic_search>apollo</semantic_search>"
        )

        assert [action.kind for action in actions] == ["exclude_docs", "semantic_search"]
        assert [action.text for action in actions] == ["2", "apollo"]

    def test_parse_doc_ids(self):
        assert parse_doc_ids("IDs: 3, 7 and 12") == [3, 7, 12]

    def test_parse_doc_ids_preserves_string_chunk_ids(self):
        assert parse_doc_ids("chunk-alpha, abc-123, alpha") == ["chunk-alpha", "abc-123", "alpha"]


class TestInteractRAGPipelineConfig:
    """Tests for InteractRAGPipelineConfig."""

    def test_shipped_yaml_uses_existing_llm_config(self):
        config = OmegaConf.load(Path("configs/pipelines/generation/interact_rag.yaml"))
        llm_name = str(config.llm)

        assert (Path("configs/llm") / f"{llm_name}.yaml").exists() or (Path("configs/llm") / f"{llm_name}.yml").exists()

    def test_get_pipeline_class(self):
        config = InteractRAGPipelineConfig(
            name="interact_rag",
            llm=FakeListLLM(responses=["<answer>ok</answer>"]),
            retrieval_pipeline_name="hybrid_rrf",
        )

        assert config.get_pipeline_class() == InteractRAGPipeline

    def test_get_pipeline_kwargs_requires_injected_retrieval_pipeline(self):
        config = InteractRAGPipelineConfig(
            name="interact_rag",
            llm=FakeListLLM(responses=["<answer>ok</answer>"]),
            retrieval_pipeline_name="hybrid_rrf",
        )

        with pytest.raises(ValueError, match="not injected"):
            config.get_pipeline_kwargs()

    def test_get_pipeline_kwargs_after_injection(self):
        retrieval_pipeline = create_mock_retrieval_pipeline(pipeline_id=77)
        llm = FakeListLLM(responses=["<answer>ok</answer>"])
        config = InteractRAGPipelineConfig(
            name="interact_rag",
            llm=llm,
            retrieval_pipeline_name="hybrid_rrf",
            max_steps=4,
            initial_scale=6,
        )

        config.inject_retrieval_pipeline(retrieval_pipeline)
        kwargs = config.get_pipeline_kwargs()

        assert kwargs["llm"] is llm
        assert kwargs["retrieval_pipeline"] is retrieval_pipeline
        assert kwargs["max_steps"] == 4
        assert kwargs["initial_scale"] == 6
        assert kwargs["exact_retrieval_pipeline"] is None

    def test_get_pipeline_kwargs_passes_exact_pipeline_name_without_injection(self):
        retrieval_pipeline = create_mock_retrieval_pipeline(pipeline_id=77)
        llm = FakeListLLM(responses=["<answer>ok</answer>"])
        config = InteractRAGPipelineConfig(
            name="interact_rag",
            llm=llm,
            retrieval_pipeline_name="vector_search",
            exact_retrieval_pipeline_name="bm25",
        )

        config.inject_retrieval_pipeline(retrieval_pipeline)
        kwargs = config.get_pipeline_kwargs()

        assert kwargs["retrieval_pipeline"] is retrieval_pipeline
        assert kwargs["exact_retrieval_pipeline"] == "bm25"


class TestInteractRAGPipeline:
    """Tests for INTERACT-RAG generation behavior."""

    def test_format_scratchpad_exposes_real_chunk_ids_and_scores(self):
        scratchpad = InteractRAGPipeline._format_scratchpad(
            ["semantic_search: apollo"],
            [{"doc_id": 42, "score": 0.75, "content": "Apollo evidence."}],
        )

        assert "[chunk_id=42 score=0.75] Apollo evidence." in scratchpad
        assert "[1]" not in scratchpad

    def test_apply_include_docs_rejects_unexposed_ordinal_ids(self):
        pipeline = _build_unit_pipeline(
            FakeListLLM(responses=["<answer>ok</answer>"]),
            create_mock_retrieval_pipeline(),
            MagicMock(),
        )
        state = InteractRAGState()
        trace: list[str] = []

        handled = pipeline._apply_state_action(
            parse_interact_rag_action("<include_docs>1</include_docs>"),
            state,
            trace,
            exposed_doc_ids={42},
        )

        assert handled is True
        assert state.included_doc_ids == []
        assert trace == ["include_docs ignored unknown IDs: 1"]

    def test_apply_include_docs_accepts_exposed_chunk_ids(self):
        pipeline = _build_unit_pipeline(
            FakeListLLM(responses=["<answer>ok</answer>"]),
            create_mock_retrieval_pipeline(),
            MagicMock(),
        )
        state = InteractRAGState()
        trace: list[str] = []

        handled = pipeline._apply_state_action(
            parse_interact_rag_action("<include_docs>42</include_docs>"),
            state,
            trace,
            exposed_doc_ids={42},
        )

        assert handled is True
        assert state.included_doc_ids == [42]
        assert trace == ["include_docs: 42"]

    def test_apply_include_docs_accepts_exposed_string_chunk_ids(self):
        pipeline = _build_unit_pipeline(
            FakeListLLM(responses=["<answer>ok</answer>"]),
            create_mock_retrieval_pipeline(),
            MagicMock(),
        )
        state = InteractRAGState()
        trace: list[str] = []

        handled = pipeline._apply_state_action(
            parse_interact_rag_action("<include_docs>chunk-alpha, abc-123, alpha</include_docs>"),
            state,
            trace,
            exposed_doc_ids={"chunk-alpha", "abc-123", "alpha"},
        )

        assert handled is True
        assert state.included_doc_ids == ["chunk-alpha", "abc-123", "alpha"]
        assert trace == ["include_docs: chunk-alpha, abc-123, alpha"]

    def test_initialization_rejects_invalid_steps(self):
        with (
            patch("autorag_research.pipelines.generation.base.BaseGenerationPipeline.__init__", return_value=None),
            pytest.raises(ValueError, match="max_steps must be >= 1"),
        ):
            InteractRAGPipeline(
                session_factory=MagicMock(),
                name="invalid_interact_rag",
                llm=FakeListLLM(responses=["<answer>ok</answer>"]),
                retrieval_pipeline=create_mock_retrieval_pipeline(),
                max_steps=0,
            )

    @pytest.mark.asyncio
    async def test_generate_runs_search_action_then_answer(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            side_effect=[
                _mock_response("<semantic_search>apollo 11 landing</semantic_search>"),
                _mock_response("<answer>Apollo 11 landed in 1969.</answer>"),
            ]
        )
        retrieval_pipeline = create_mock_retrieval_pipeline(
            default_results=[{"doc_id": 10, "score": 0.9, "content": "Apollo 11 landed in July 1969."}]
        )
        service = MagicMock()
        service.get_query_text.return_value = "When did Apollo 11 land?"
        pipeline = _build_unit_pipeline(llm, retrieval_pipeline, service, max_steps=3, initial_scale=2)

        result = await pipeline._generate(1, top_k=2)

        retrieval_pipeline.retrieve.assert_awaited_once_with("apollo 11 landing", 2)
        assert result.metadata is not None
        assert result.text == "Apollo 11 landed in 1969."
        assert result.metadata["retrieved_chunk_ids"] == [10]
        assert result.metadata["evidence"] == ["Apollo 11 landed in July 1969."]
        assert result.metadata["terminated_by"] == "answer"
        assert result.token_usage == {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10}
        assert result.metadata["exact_engine"] is False

    @pytest.mark.asyncio
    async def test_generate_applies_scale_and_exclude_controls(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            side_effect=[
                _mock_response("<adjust_scale>4</adjust_scale>"),
                _mock_response("<exact_search>target terms</exact_search>"),
                _mock_response("<exclude_docs>2</exclude_docs>"),
                _mock_response("<exact_search>target terms</exact_search>"),
                _mock_response("fallback answer"),
            ]
        )
        retrieval_pipeline = create_mock_retrieval_pipeline(
            default_results=[
                {"doc_id": 2, "score": 0.9, "content": "excluded"},
                {"doc_id": 3, "score": 0.8, "content": "kept"},
            ]
        )
        service = MagicMock()
        service.get_query_text.return_value = "Question?"
        pipeline = _build_unit_pipeline(
            llm,
            retrieval_pipeline,
            service,
            max_steps=4,
            initial_scale=1,
            max_scale=5,
        )

        result = await pipeline._generate(1, top_k=1)

        assert result.metadata is not None
        assert retrieval_pipeline.retrieve.await_args_list[0].args == ("exact: target terms", 4)
        assert retrieval_pipeline.retrieve.await_args_list[1].args == ("exact: target terms", 4)
        assert result.metadata["excluded_doc_ids"] == [2]
        assert result.metadata["retrieved_chunk_ids"] == [3]
        assert result.metadata["evidence"] == ["kept"]
        assert result.metadata["terminated_by"] == "max_steps_fallback"

    @pytest.mark.asyncio
    async def test_exact_search_routes_to_exact_pipeline_with_raw_keywords(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            side_effect=[
                _mock_response("<exact_search>target terms</exact_search>"),
                _mock_response("<answer>done</answer>"),
            ]
        )
        retrieval_pipeline = create_mock_retrieval_pipeline(default_results=[])
        exact_pipeline = create_mock_retrieval_pipeline(
            default_results=[{"doc_id": 11, "score": 1.0, "content": "exact evidence"}]
        )
        service = MagicMock()
        service.get_query_text.return_value = "Question?"
        pipeline = _build_unit_pipeline(
            llm, retrieval_pipeline, service, max_steps=2, initial_scale=4, exact_retrieval_pipeline=exact_pipeline
        )

        result = await pipeline._generate(1, top_k=4)

        retrieval_pipeline.retrieve.assert_not_awaited()
        exact_pipeline.retrieve.assert_awaited_once_with("target terms", 4)
        assert result.metadata["evidence"] == ["exact evidence"]
        assert result.metadata["retrieval_action_mode"] == "engine"
        assert result.metadata["exact_engine"] is True
        assert result.metadata["degraded_actions"] == []

    @pytest.mark.asyncio
    async def test_entity_match_uses_exact_pipeline_and_returns_top_three_snippets(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            side_effect=[
                _mock_response("<entity_match>Marie Curie</entity_match>"),
                _mock_response("<answer>done</answer>"),
            ]
        )
        retrieval_pipeline = create_mock_retrieval_pipeline(default_results=[])
        exact_pipeline = create_mock_retrieval_pipeline(
            default_results=[
                {"doc_id": 1, "score": 0.9, "content": "one"},
                {"doc_id": 2, "score": 0.8, "content": "two"},
                {"doc_id": 3, "score": 0.7, "content": "three"},
                {"doc_id": 4, "score": 0.6, "content": "four"},
            ]
        )
        service = MagicMock()
        service.get_query_text.return_value = "Question?"
        pipeline = _build_unit_pipeline(
            llm, retrieval_pipeline, service, max_steps=2, initial_scale=5, exact_retrieval_pipeline=exact_pipeline
        )

        result = await pipeline._generate(1, top_k=5)

        exact_pipeline.retrieve.assert_awaited_once_with("Marie Curie", 5)
        assert result.metadata["retrieved_chunk_ids"] == [1, 2, 3]
        assert result.metadata["evidence"] == ["one", "two", "three"]

    @pytest.mark.asyncio
    async def test_weighted_fusion_normalizes_and_fuses_both_engines(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            side_effect=[
                _mock_response('<weighted_fusion semantic="0.25" exact="0.75">apollo</weighted_fusion>'),
                _mock_response("<answer>done</answer>"),
            ]
        )
        retrieval_pipeline = create_mock_retrieval_pipeline()
        retrieval_pipeline.retrieve = AsyncMock(
            return_value=[
                {"doc_id": "a", "score": 0.9, "content": "semantic best"},
                {"doc_id": "b", "score": 0.5, "content": "shared"},
                {"doc_id": "c", "score": 0.1, "content": "semantic worst"},
            ]
        )
        exact_pipeline = create_mock_retrieval_pipeline()
        exact_pipeline.retrieve = AsyncMock(
            return_value=[
                {"doc_id": "c", "score": 100.0, "content": "semantic worst"},
                {"doc_id": "b", "score": 60.0, "content": "shared"},
                {"doc_id": "a", "score": 20.0, "content": "semantic best"},
            ]
        )
        service = MagicMock()
        service.get_query_text.return_value = "Question?"
        pipeline = _build_unit_pipeline(
            llm, retrieval_pipeline, service, max_steps=2, initial_scale=3, exact_retrieval_pipeline=exact_pipeline
        )

        result = await pipeline._generate(1, top_k=3)

        retrieval_pipeline.retrieve.assert_awaited_once_with("apollo", 20)
        exact_pipeline.retrieve.assert_awaited_once_with("apollo", 20)
        assert result.metadata["retrieved_chunk_ids"] == ["c", "b", "a"]
        assert result.metadata["semantic_weight"] == 0.25
        assert result.metadata["exact_weight"] == 0.75

    @pytest.mark.asyncio
    async def test_multi_action_applies_exclude_before_same_step_semantic_retrieval(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            side_effect=[
                _mock_response("<semantic_search>first</semantic_search>"),
                _mock_response("<exclude_docs>2</exclude_docs><semantic_search>second</semantic_search>"),
                _mock_response("<answer>done</answer>"),
            ]
        )
        retrieval_pipeline = create_mock_retrieval_pipeline()
        retrieval_pipeline.retrieve = AsyncMock(
            side_effect=[
                [
                    {"doc_id": 2, "score": 0.9, "content": "excluded"},
                    {"doc_id": 3, "score": 0.8, "content": "kept"},
                ],
                [
                    {"doc_id": 2, "score": 0.95, "content": "excluded again"},
                    {"doc_id": 4, "score": 0.7, "content": "new kept"},
                ],
            ]
        )
        service = MagicMock()
        service.get_query_text.return_value = "Question?"
        pipeline = _build_unit_pipeline(llm, retrieval_pipeline, service, max_steps=3, initial_scale=2)

        result = await pipeline._generate(1, top_k=2)

        assert result.metadata["excluded_doc_ids"] == [2]
        assert result.metadata["retrieved_chunk_ids"] == [3, 4]
        assert result.metadata["evidence"] == ["kept", "new kept"]
        assert "exclude_docs: 2" in result.metadata["trace"]
        assert "semantic_search: second" in result.metadata["trace"]

    @pytest.mark.asyncio
    async def test_answer_with_same_step_state_action_applies_state_then_terminates(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            side_effect=[
                _mock_response("<semantic_search>first</semantic_search>"),
                _mock_response(
                    "<exclude_docs>2</exclude_docs><semantic_search>moot</semantic_search><answer>done</answer>"
                ),
            ]
        )
        retrieval_pipeline = create_mock_retrieval_pipeline()
        retrieval_pipeline.retrieve = AsyncMock(
            return_value=[
                {"doc_id": 2, "score": 0.9, "content": "excluded"},
                {"doc_id": 3, "score": 0.8, "content": "kept"},
            ]
        )
        service = MagicMock()
        service.get_query_text.return_value = "Question?"
        pipeline = _build_unit_pipeline(llm, retrieval_pipeline, service, max_steps=3, initial_scale=2)

        result = await pipeline._generate(1, top_k=2)

        assert result.text == "done"
        assert result.metadata["terminated_by"] == "answer"
        assert result.metadata["excluded_doc_ids"] == [2]
        assert result.metadata["evidence"] == ["kept"]
        assert retrieval_pipeline.retrieve.await_count == 1
        assert any("skipped same-step retrieval actions" in entry for entry in result.metadata["trace"])

    @pytest.mark.asyncio
    async def test_generate_applies_include_docs_for_exposed_ids_and_backfills_content(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            side_effect=[
                _mock_response("<semantic_search>Curie</semantic_search>"),
                _mock_response("<include_docs>5</include_docs>"),
                _mock_response("<entity_match>Curie</entity_match>"),
                _mock_response("<answer>Curie answer.</answer>"),
            ]
        )
        retrieval_pipeline = create_mock_retrieval_pipeline(
            default_results=[{"doc_id": 5, "score": 0.9}, {"doc_id": 6, "score": 0.8}]
        )
        service = MagicMock()
        service.get_query_text.return_value = "Question?"
        service.get_chunk_contents.return_value = ["Forced evidence.", "Fetched evidence."]
        pipeline = _build_unit_pipeline(llm, retrieval_pipeline, service, max_steps=4, initial_scale=2)

        result = await pipeline._generate(1, top_k=2)

        assert result.metadata is not None
        assert retrieval_pipeline.retrieve.await_args_list[0].args == ("Curie", 2)
        assert retrieval_pipeline.retrieve.await_args_list[1].args == ("entity: Curie", 2)
        service.get_chunk_contents.assert_any_call([5, 6])
        assert result.metadata["included_doc_ids"] == [5]
        assert result.metadata["retrieved_chunk_ids"] == [5, 6]
        assert result.metadata["evidence"] == ["Forced evidence.", "Fetched evidence."]
        second_prompt = llm.ainvoke.await_args_list[1].args[0]
        assert "[chunk_id=5 score=0.9] Forced evidence." in second_prompt

    @pytest.mark.asyncio
    async def test_generate_records_prompt_simulated_degradation_for_weighted_fusion(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            side_effect=[
                _mock_response('<weighted_fusion semantic="0.8" exact="0.2">apollo</weighted_fusion>'),
                _mock_response("<answer>done</answer>"),
            ]
        )
        retrieval_pipeline = create_mock_retrieval_pipeline(
            default_results=[{"doc_id": 10, "score": 0.9, "content": "Apollo evidence."}]
        )
        service = MagicMock()
        service.get_query_text.return_value = "Question?"
        pipeline = _build_unit_pipeline(llm, retrieval_pipeline, service, max_steps=2, initial_scale=2)

        result = await pipeline._generate(1, top_k=2)

        assert result.metadata is not None
        retrieval_pipeline.retrieve.assert_awaited_once_with(
            "semantic weight 0.80; exact weight 0.20; query: apollo", 2
        )
        assert result.metadata["retrieval_action_mode"] == "prompt_simulated"
        assert result.metadata["degraded_actions"] == ["weighted_fusion"]
        assert "degraded weighted_fusion" in result.metadata["trace"][0]

    @pytest.mark.asyncio
    async def test_generate_keeps_included_doc_within_evidence_budget(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            side_effect=[
                _mock_response("<semantic_search>first</semantic_search>"),
                _mock_response("<include_docs>5</include_docs>"),
                _mock_response("<semantic_search>second</semantic_search>"),
                _mock_response("<answer>done</answer>"),
            ]
        )
        retrieval_pipeline = create_mock_retrieval_pipeline()
        retrieval_pipeline.retrieve = AsyncMock(
            side_effect=[
                [{"doc_id": 5, "score": 0.9, "content": "pinned"}],
                [
                    {"doc_id": 6, "score": 0.8, "content": "new6"},
                    {"doc_id": 7, "score": 0.7, "content": "new7"},
                    {"doc_id": 8, "score": 0.6, "content": "new8"},
                ],
            ]
        )
        service = MagicMock()
        service.get_query_text.return_value = "Question?"
        pipeline = _build_unit_pipeline(
            llm, retrieval_pipeline, service, max_steps=4, initial_scale=3, evidence_budget=2
        )

        result = await pipeline._generate(1, top_k=3)

        assert result.metadata is not None
        assert result.metadata["included_doc_ids"] == [5]
        assert result.metadata["evidence"] == ["pinned", "new8"]

    @pytest.mark.asyncio
    async def test_generate_applies_string_include_and_exclude_controls(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            side_effect=[
                _mock_response("<semantic_search>first</semantic_search>"),
                _mock_response("<include_docs>chunk-alpha</include_docs>"),
                _mock_response("<exclude_docs>abc-123</exclude_docs>"),
                _mock_response("<semantic_search>second</semantic_search>"),
                _mock_response("<answer>done</answer>"),
            ]
        )
        retrieval_pipeline = create_mock_retrieval_pipeline(
            default_results=[
                {"doc_id": "chunk-alpha", "score": 0.9, "content": "included"},
                {"doc_id": "abc-123", "score": 0.8, "content": "excluded"},
            ]
        )
        service = MagicMock()
        service.get_query_text.return_value = "Question?"
        pipeline = _build_unit_pipeline(llm, retrieval_pipeline, service, max_steps=5, initial_scale=2)

        result = await pipeline._generate("query-alpha", top_k=2)

        assert result.metadata is not None
        assert result.metadata["included_doc_ids"] == ["chunk-alpha"]
        assert result.metadata["excluded_doc_ids"] == ["abc-123"]
        assert result.metadata["retrieved_chunk_ids"] == ["chunk-alpha"]
        assert result.metadata["evidence"] == ["included"]
