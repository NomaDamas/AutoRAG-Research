"""Tests for the DeepRAG generation pipeline."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hydra.utils import instantiate
from langchain_core.language_models.fake import FakeListLLM
from omegaconf import OmegaConf

from autorag_research.pipelines.generation.deep_rag import DeepRAGPipeline, DeepRAGPipelineConfig, parse_deeprag_action
from tests.autorag_research.pipelines.pipeline_test_utils import create_mock_retrieval_pipeline


def _mock_response(content: str) -> MagicMock:
    response = MagicMock()
    response.content = content
    response.usage_metadata = {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}
    return response


def _build_unit_pipeline(llm: Any, retrieval_pipeline: Any, service: Any, **kwargs: Any) -> DeepRAGPipeline:
    with patch("autorag_research.pipelines.generation.base.BaseGenerationPipeline.__init__", return_value=None):
        pipeline = DeepRAGPipeline(
            session_factory=MagicMock(),
            name="deeprag_unit",
            llm=llm,
            retrieval_pipeline=retrieval_pipeline,
            **kwargs,
        )
    pipeline._llm = llm
    pipeline._retrieval_pipeline = retrieval_pipeline
    pipeline._service = service
    pipeline.pipeline_id = 1
    return pipeline


class TestDeepRAGActionParsing:
    """Tests for DeepRAG action parsing."""

    def test_parse_retrieve(self):
        action = parse_deeprag_action("Need evidence. <retrieve>moon origin</retrieve>")

        assert action.kind == "retrieve"
        assert action.text == "moon origin"

    def test_parse_parametric(self):
        action = parse_deeprag_action("<parametric>known historical fact</parametric>")

        assert action.kind == "parametric"
        assert action.text == "known historical fact"

    def test_parse_answer_preferred(self):
        action = parse_deeprag_action("<retrieve>x</retrieve><answer>final</answer>")

        assert action.kind == "answer"
        assert action.text == "final"

    def test_parse_answer_prefix(self):
        action = parse_deeprag_action("answer: final")

        assert action.kind == "answer"
        assert action.text == "final"

    def test_parse_untagged_text_as_invalid(self):
        action = parse_deeprag_action("standalone but untagged subquery")

        assert action.kind == "invalid"
        assert action.text == "standalone but untagged subquery"


class TestDeepRAGPipelineConfig:
    """Tests for DeepRAGPipelineConfig."""

    def test_shipped_yaml_config_instantiates(self):
        cfg = OmegaConf.load("configs/pipelines/generation/deep_rag.yaml")

        config = instantiate(cfg)

        assert isinstance(config, DeepRAGPipelineConfig)
        assert config.name == "deep_rag"
        assert config.retrieval_pipeline_name == "bm25"

    def test_get_pipeline_class(self):
        config = DeepRAGPipelineConfig(
            name="deep_rag",
            llm=FakeListLLM(responses=["<answer>ok</answer>"]),
            retrieval_pipeline_name="bm25",
        )

        assert config.get_pipeline_class() == DeepRAGPipeline

    def test_get_pipeline_kwargs_requires_injected_retrieval_pipeline(self):
        config = DeepRAGPipelineConfig(
            name="deep_rag",
            llm=FakeListLLM(responses=["<answer>ok</answer>"]),
            retrieval_pipeline_name="bm25",
        )

        with pytest.raises(ValueError, match="not injected"):
            config.get_pipeline_kwargs()

    def test_get_pipeline_kwargs_after_injection(self):
        retrieval_pipeline = create_mock_retrieval_pipeline(pipeline_id=33)
        llm = FakeListLLM(responses=["<answer>ok</answer>"])
        config = DeepRAGPipelineConfig(
            name="deep_rag",
            llm=llm,
            retrieval_pipeline_name="bm25",
            max_steps=4,
            k_per_retrieval=3,
        )

        config.inject_retrieval_pipeline(retrieval_pipeline)
        kwargs = config.get_pipeline_kwargs()

        assert kwargs["llm"] is llm
        assert kwargs["retrieval_pipeline"] is retrieval_pipeline
        assert kwargs["max_steps"] == 4
        assert kwargs["k_per_retrieval"] == 3
        assert "intermediate_prompt_template" in kwargs


class TestDeepRAGPipeline:
    """Tests for DeepRAG generation behavior."""

    def test_initialization_rejects_invalid_steps(self):
        with (
            patch("autorag_research.pipelines.generation.base.BaseGenerationPipeline.__init__", return_value=None),
            pytest.raises(ValueError, match="max_steps must be >= 1"),
        ):
            DeepRAGPipeline(
                session_factory=MagicMock(),
                name="invalid_deeprag",
                llm=FakeListLLM(responses=["<answer>ok</answer>"]),
                retrieval_pipeline=create_mock_retrieval_pipeline(),
                max_steps=0,
            )

    @pytest.mark.asyncio
    async def test_generate_retrieve_parametric_intermediates_then_answers(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            side_effect=[
                _mock_response("<retrieve>q1</retrieve>"),
                _mock_response("ia1"),
                _mock_response("<parametric>q2</parametric>"),
                _mock_response("ia2"),
                _mock_response("<answer>final</answer>"),
            ]
        )
        retrieval_pipeline = create_mock_retrieval_pipeline(
            default_results=[{"doc_id": 10, "score": 0.9, "content": "passage for q1"}]
        )
        service = MagicMock()
        service.get_query_text.return_value = "Question?"
        pipeline = _build_unit_pipeline(llm, retrieval_pipeline, service, max_steps=5, k_per_retrieval=3)

        result = await pipeline._generate(1, top_k=2)

        retrieval_pipeline.retrieve.assert_awaited_once_with("q1", 2)
        prompts = [call.args[0] for call in llm.ainvoke.await_args_list]
        assert prompts[1].count("Retrieved passages:") == 1
        assert "passage for q1" in prompts[1]
        assert "Retrieved passages:" not in prompts[3]
        assert "Follow up: q1" in prompts[2]
        assert "Context:" in prompts[2]
        assert "passage for q1" in prompts[2]
        assert "Intermediate answer: ia1" in prompts[2]
        assert result.text == "final"
        assert result.metadata is not None
        assert result.metadata["follow_up_queries"] == ["q1", "q2"]
        assert result.metadata["retrieved_chunk_ids"] == [10]
        assert result.metadata["retrieved_scores"] == [0.9]
        assert result.metadata["trajectory"] == [
            {
                "step": 1,
                "subquery": "q1",
                "decision": "retrieve",
                "retrieved_chunk_ids": [10],
                "retrieved_passages": ["passage for q1"],
                "intermediate_answer": "ia1",
            },
            {
                "step": 2,
                "subquery": "q2",
                "decision": "parametric",
                "retrieved_chunk_ids": [],
                "retrieved_passages": [],
                "intermediate_answer": "ia2",
            },
        ]
        assert result.metadata["terminated_by"] == "answer"
        assert result.token_usage == {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25}

    @pytest.mark.asyncio
    async def test_generate_uses_k_per_retrieval_when_top_k_is_not_positive(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            side_effect=[
                _mock_response("<retrieve>broad evidence</retrieve>"),
                _mock_response("intermediate"),
                _mock_response("<answer>done</answer>"),
            ]
        )
        retrieval_pipeline = create_mock_retrieval_pipeline(
            default_results=[{"doc_id": 11, "score": 0.8, "content": "Evidence."}]
        )
        service = MagicMock()
        service.get_query_text.return_value = "Question?"
        pipeline = _build_unit_pipeline(llm, retrieval_pipeline, service, max_steps=2, k_per_retrieval=4)

        result = await pipeline._generate(1, top_k=0)

        retrieval_pipeline.retrieve.assert_awaited_once_with("broad evidence", 4)
        assert result.metadata is not None
        assert result.metadata["effective_retrieval_k"] == 4
        assert result.metadata["top_k_cap"] == 0

    @pytest.mark.asyncio
    async def test_generate_falls_back_after_max_steps_with_trajectory(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            side_effect=[
                _mock_response("<parametric>q1</parametric>"),
                _mock_response("ia1"),
                _mock_response("fallback answer"),
            ]
        )
        service = MagicMock()
        service.get_query_text.return_value = "Question?"
        pipeline = _build_unit_pipeline(llm, create_mock_retrieval_pipeline(), service, max_steps=1)

        result = await pipeline._generate(1, top_k=5)

        final_prompt = llm.ainvoke.await_args_list[-1].args[0]
        assert "Follow up: q1" in final_prompt
        assert "Intermediate answer: ia1" in final_prompt
        assert result.text == "fallback answer"
        assert result.metadata is not None
        assert result.metadata["trajectory"][0]["subquery"] == "q1"
        assert result.metadata["terminated_by"] == "max_steps_fallback"

    @pytest.mark.asyncio
    async def test_generate_invalid_controller_output_records_note_then_continues(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            side_effect=[
                _mock_response("untagged subquery"),
                _mock_response("<answer>done</answer>"),
            ]
        )
        service = MagicMock()
        service.get_query_text.return_value = "Question?"
        retrieval_pipeline = create_mock_retrieval_pipeline()
        pipeline = _build_unit_pipeline(llm, retrieval_pipeline, service, max_steps=2)

        result = await pipeline._generate(1, top_k=1)

        retrieval_pipeline.retrieve.assert_not_awaited()
        assert result.text == "done"
        assert result.metadata is not None
        assert result.metadata["trajectory"][0] == {
            "step": 1,
            "subquery": "(invalid controller output)",
            "decision": "parametric",
            "retrieved_chunk_ids": [],
            "retrieved_passages": [],
            "intermediate_answer": "Invalid controller output ignored: untagged subquery",
        }

    @pytest.mark.asyncio
    async def test_generate_backfills_missing_contents(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            side_effect=[
                _mock_response("<retrieve>missing content</retrieve>"),
                _mock_response("intermediate"),
                _mock_response("<answer>done</answer>"),
            ]
        )
        retrieval_pipeline = create_mock_retrieval_pipeline(default_results=[{"doc_id": 5, "score": 0.7}])
        service = MagicMock()
        service.get_query_text.return_value = "Question?"
        service.get_chunk_contents.return_value = ["Fetched evidence."]
        pipeline = _build_unit_pipeline(llm, retrieval_pipeline, service, max_steps=2, k_per_retrieval=1)

        result = await pipeline._generate(1, top_k=1)

        service.get_chunk_contents.assert_called_once_with([5])
        assert result.metadata is not None
        assert result.metadata["evidence"] == ["Fetched evidence."]
        assert result.metadata["trajectory"][0]["intermediate_answer"] == "intermediate"
