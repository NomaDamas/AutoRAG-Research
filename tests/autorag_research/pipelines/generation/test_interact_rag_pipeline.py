"""Tests for the INTERACT-RAG generation pipeline."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.language_models.fake import FakeListLLM

from autorag_research.pipelines.generation.interact_rag import (
    InteractRAGPipeline,
    InteractRAGPipelineConfig,
    parse_doc_ids,
    parse_interact_rag_action,
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
    return pipeline


class TestInteractRAGParsing:
    """Tests for action parsing helpers."""

    def test_parse_semantic_search(self):
        action = parse_interact_rag_action("<semantic_search>moon landing timeline</semantic_search>")

        assert action.kind == "semantic_search"
        assert action.text == "moon landing timeline"

    def test_parse_weighted_fusion_with_weights(self):
        action = parse_interact_rag_action("<weighted_fusion semantic=\"0.7\" exact=\"0.3\">apollo 11</weighted_fusion>")

        assert action.kind == "weighted_fusion"
        assert action.text == "apollo 11"
        assert action.semantic_weight == 0.7
        assert action.exact_weight == 0.3

    def test_parse_answer_preferred(self):
        action = parse_interact_rag_action("<semantic_search>x</semantic_search><answer>final</answer>")

        assert action.kind == "answer"
        assert action.text == "final"

    def test_parse_doc_ids(self):
        assert parse_doc_ids("IDs: 3, 7 and 12") == [3, 7, 12]


class TestInteractRAGPipelineConfig:
    """Tests for InteractRAGPipelineConfig."""

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


class TestInteractRAGPipeline:
    """Tests for INTERACT-RAG generation behavior."""

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
        assert result.text == "Apollo 11 landed in 1969."
        assert result.metadata["retrieved_chunk_ids"] == [10]
        assert result.metadata["evidence"] == ["Apollo 11 landed in July 1969."]
        assert result.metadata["terminated_by"] == "answer"
        assert result.token_usage == {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10}

    @pytest.mark.asyncio
    async def test_generate_applies_scale_and_exclude_controls(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            side_effect=[
                _mock_response("<adjust_scale>4</adjust_scale>"),
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
            max_steps=3,
            initial_scale=1,
            max_scale=5,
        )

        result = await pipeline._generate(1, top_k=1)

        retrieval_pipeline.retrieve.assert_awaited_once_with("exact: target terms", 4)
        assert result.metadata["excluded_doc_ids"] == [2]
        assert result.metadata["retrieved_chunk_ids"] == [3]
        assert result.metadata["evidence"] == ["kept"]
        assert result.metadata["terminated_by"] == "max_steps_fallback"

    @pytest.mark.asyncio
    async def test_generate_applies_include_docs_and_backfills_content(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            side_effect=[
                _mock_response("<include_docs>5</include_docs>"),
                _mock_response("<entity_match>Curie</entity_match>"),
                _mock_response("<answer>Curie answer.</answer>"),
            ]
        )
        retrieval_pipeline = create_mock_retrieval_pipeline(default_results=[{"doc_id": 6, "score": 0.8}])
        service = MagicMock()
        service.get_query_text.return_value = "Question?"
        service.get_chunk_contents.return_value = ["Forced evidence.", "Fetched evidence."]
        pipeline = _build_unit_pipeline(llm, retrieval_pipeline, service, max_steps=3, initial_scale=2)

        result = await pipeline._generate(1, top_k=2)

        retrieval_pipeline.retrieve.assert_awaited_once_with("entity: Curie", 2)
        service.get_chunk_contents.assert_called_once_with([5, 6])
        assert result.metadata["included_doc_ids"] == [5]
        assert result.metadata["retrieved_chunk_ids"] == [5, 6]
        assert result.metadata["evidence"] == ["Forced evidence.", "Fetched evidence."]
