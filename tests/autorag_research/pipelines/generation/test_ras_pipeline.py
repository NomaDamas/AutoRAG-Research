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
        action = parse_ras_plan_action("<subquery>apollo 11 crew</subquery>")

        assert action.kind == "retrieve"
        assert action.text == "apollo 11 crew"

    def test_parse_sufficient_action(self):
        action = parse_ras_plan_action("<sufficient>graph is enough</sufficient>")

        assert action.kind == "sufficient"
        assert action.text == "graph is enough"

    def test_parse_malformed_action_is_invalid(self):
        action = parse_ras_plan_action("I need Apollo 11 context but forgot the tags.")

        assert action.kind == "invalid"
        assert action.text == "I need Apollo 11 context but forgot the tags."

    def test_parse_untagged_subquery_prefix_is_invalid(self):
        action = parse_ras_plan_action("subquery: apollo 11 crew")

        assert action.kind == "invalid"
        assert action.text == "subquery: apollo 11 crew"

    def test_parse_empty_required_tags_are_invalid(self):
        assert parse_ras_plan_action("<subquery> </subquery>").kind == "invalid"
        assert parse_ras_plan_action("<sufficient> </sufficient>").kind == "invalid"

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
    async def test_generate_retrieves_extracts_triples_and_answers(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            side_effect=[
                _mock_response("<triple>Apollo 11 | mission | lunar landing</triple>"),
                _mock_response("<subquery>apollo 11 crew</subquery>"),
                _mock_response("<triple>Apollo 11 | commander | Neil Armstrong</triple>"),
                _mock_response("<sufficient>enough</sufficient>"),
                _mock_response("Neil Armstrong commanded Apollo 11."),
            ]
        )
        retrieval_pipeline = create_mock_retrieval_pipeline()
        retrieval_pipeline._retrieve_by_id = AsyncMock(
            return_value=[{"doc_id": 9, "score": 1.0, "content": "Apollo 11 was a lunar landing mission."}]
        )
        retrieval_pipeline.retrieve = AsyncMock(
            return_value=[{"doc_id": 10, "score": 0.9, "content": "Neil Armstrong commanded Apollo 11."}]
        )
        service = MagicMock()
        service.get_query_text.return_value = "Who commanded Apollo 11?"
        pipeline = _build_unit_pipeline(llm, retrieval_pipeline, service, max_steps=3, k_per_step=2)

        result = await pipeline._generate(1, top_k=2)
        metadata = _metadata(result)

        retrieval_pipeline._retrieve_by_id.assert_awaited_once_with(1, 2)
        retrieval_pipeline.retrieve.assert_awaited_once_with("apollo 11 crew", 2)
        assert result.text == "Neil Armstrong commanded Apollo 11."
        assert metadata["subqueries"] == ["apollo 11 crew"]
        assert metadata["retrieved_chunk_ids"] == [9, 10]
        assert metadata["triples"] == [
            ("Apollo 11", "mission", "lunar landing"),
            ("Apollo 11", "commander", "Neil Armstrong"),
        ]
        assert metadata["malformed_plans"] == []
        assert result.token_usage == {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25}

    @pytest.mark.asyncio
    async def test_generate_skips_duplicate_passages_before_extracting_triples(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            side_effect=[
                _mock_response("<triple>Apollo 11 | mission | lunar landing</triple>"),
                _mock_response("<subquery>apollo 11 crew</subquery>"),
                _mock_response("Neil Armstrong commanded Apollo 11."),
            ]
        )
        retrieval_pipeline = create_mock_retrieval_pipeline()
        retrieval_pipeline._retrieve_by_id = AsyncMock(
            return_value=[{"doc_id": 9, "score": 1.0, "content": "Apollo 11 was a lunar landing mission."}]
        )
        retrieval_pipeline.retrieve = AsyncMock(
            return_value=[{"doc_id": 9, "score": 0.9, "content": "Apollo 11 was a lunar landing mission."}]
        )
        service = MagicMock()
        service.get_query_text.return_value = "Who commanded Apollo 11?"
        pipeline = _build_unit_pipeline(llm, retrieval_pipeline, service, max_steps=2, k_per_step=1)

        result = await pipeline._generate(1, top_k=1)
        metadata = _metadata(result)

        assert llm.ainvoke.await_count == 3
        retrieval_pipeline.retrieve.assert_awaited_once_with("apollo 11 crew", 1)
        assert metadata["retrieved_chunk_ids"] == [9]
        assert metadata["evidence"] == ["Apollo 11 was a lunar landing mission."]
        assert metadata["triples"] == [("Apollo 11", "mission", "lunar landing")]

    @pytest.mark.asyncio
    async def test_generate_backfills_missing_content(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            side_effect=[
                _mock_response("<subquery>missing content</subquery>"),
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
    async def test_generate_uses_original_query_fallback_for_malformed_plan(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            side_effect=[
                _mock_response("Need more evidence but omitted required tags."),
                _mock_response("<triple>Apollo 11 | commander | Neil Armstrong</triple>"),
                _mock_response("Neil Armstrong."),
            ]
        )
        retrieval_pipeline = create_mock_retrieval_pipeline()
        retrieval_pipeline._retrieve_by_id = AsyncMock(return_value=[])
        retrieval_pipeline.retrieve = AsyncMock(
            return_value=[{"doc_id": 10, "score": 0.9, "content": "Neil Armstrong commanded Apollo 11."}]
        )
        service = MagicMock()
        service.get_query_text.return_value = "Who commanded Apollo 11?"
        pipeline = _build_unit_pipeline(llm, retrieval_pipeline, service, max_steps=1, k_per_step=2)

        result = await pipeline._generate(1, top_k=2)
        metadata = _metadata(result)

        retrieval_pipeline.retrieve.assert_awaited_once_with("Who commanded Apollo 11?", 2)
        assert metadata["subqueries"] == ["Who commanded Apollo 11?"]
        assert metadata["malformed_plans"] == ["Need more evidence but omitted required tags."]
        assert metadata["plan_trace"] == ["invalid: Need more evidence but omitted required tags."]

    def test_merge_triples_deduplicates(self):
        pipeline = _build_unit_pipeline(MagicMock(), create_mock_retrieval_pipeline(), MagicMock())
        triple = RASTriple("A", "p", "B")

        assert pipeline._merge_triples([triple], [triple]) == [triple]
