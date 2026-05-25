from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from langchain_core.language_models.fake import FakeListLLM

from autorag_research.pipelines.generation.adaptive_rag import AdaptiveRAGPipeline, AdaptiveRAGPipelineConfig
from tests.autorag_research.pipelines.pipeline_test_utils import (
    PipelineTestConfig,
    PipelineTestVerifier,
    cleanup_pipeline_results_factory,
    create_mock_llm,
    create_mock_retrieval_pipeline,
)


@pytest.fixture
def cleanup(session_factory):
    yield from cleanup_pipeline_results_factory(session_factory)


@pytest.fixture
def mock_retrieval():
    return create_mock_retrieval_pipeline(
        default_results=[
            {"doc_id": 1, "score": 0.95},
            {"doc_id": 2, "score": 0.85},
            {"doc_id": 3, "score": 0.75},
        ]
    )


class TestAdaptiveRAGPipeline:
    @pytest.mark.asyncio
    async def test_zero_route_skips_retrieval(self, session_factory, mock_retrieval, cleanup):
        llm = FakeListLLM(responses=["simple", "Direct answer"])
        pipeline = AdaptiveRAGPipeline(
            session_factory=session_factory,
            name="test_adaptive_rag_zero",
            llm=llm,
            retrieval_pipeline=mock_retrieval,
        )
        cleanup.append(pipeline.pipeline_id)

        result = await pipeline._generate(query_id=1, top_k=3)

        assert result.text == "Direct answer"
        assert result.metadata is not None
        assert result.metadata["complexity_tier"] == "simple"
        assert result.metadata["route"] == "zero"
        assert result.metadata["retrieved_chunk_ids"] == []
        assert result.metadata["follow_up_queries"] == []
        assert mock_retrieval._retrieve_by_id.await_count == 0
        assert mock_retrieval.retrieve.await_count == 0

    @pytest.mark.asyncio
    async def test_single_route_uses_single_retrieval(self, session_factory, mock_retrieval, cleanup):
        llm = FakeListLLM(responses=["moderate", "Single answer"])
        pipeline = AdaptiveRAGPipeline(
            session_factory=session_factory,
            name="test_adaptive_rag_single",
            llm=llm,
            retrieval_pipeline=mock_retrieval,
        )
        cleanup.append(pipeline.pipeline_id)

        result = await pipeline._generate(query_id=1, top_k=3)

        assert result.text == "Single answer"
        assert result.metadata is not None
        assert result.metadata["complexity_tier"] == "moderate"
        assert result.metadata["route"] == "single"
        assert result.metadata["retrieved_chunk_ids"] == [1, 2, 3]
        assert result.metadata["retrieved_scores"] == [0.95, 0.85, 0.75]
        assert result.metadata["follow_up_queries"] == []
        mock_retrieval._retrieve_by_id.assert_awaited_once_with(1, 3)
        assert mock_retrieval.retrieve.await_count == 0

    @pytest.mark.asyncio
    async def test_multi_route_performs_iterative_retrieval(self, session_factory, mock_retrieval, cleanup):
        llm = FakeListLLM(responses=["complex", "follow up query", "STOP", "Complex answer"])
        mock_retrieval._retrieve_by_id = AsyncMock(
            return_value=[{"doc_id": 1, "score": 0.8}, {"doc_id": 2, "score": 0.7}]
        )
        mock_retrieval.retrieve = AsyncMock(return_value=[{"doc_id": 4, "score": 0.95}])

        pipeline = AdaptiveRAGPipeline(
            session_factory=session_factory,
            name="test_adaptive_rag_multi",
            llm=llm,
            retrieval_pipeline=mock_retrieval,
            max_multi_steps=2,
        )
        cleanup.append(pipeline.pipeline_id)

        result = await pipeline._generate(query_id=1, top_k=4)

        assert result.text == "Complex answer"
        assert result.metadata is not None
        assert result.metadata["complexity_tier"] == "complex"
        assert result.metadata["route"] == "multi"
        assert result.metadata["follow_up_queries"] == ["follow up query"]
        assert result.metadata["retrieved_chunk_ids"] == [4, 1, 2]
        assert result.metadata["retrieved_scores"] == [0.95, 0.8, 0.7]
        mock_retrieval._retrieve_by_id.assert_awaited_once_with(1, 4)
        mock_retrieval.retrieve.assert_awaited_once_with("follow up query", 4)

    @pytest.mark.asyncio
    async def test_unknown_complexity_falls_back_to_single_route(self, session_factory, mock_retrieval, cleanup):
        llm = FakeListLLM(responses=["uncertain", "Fallback single answer"])
        pipeline = AdaptiveRAGPipeline(
            session_factory=session_factory,
            name="test_adaptive_rag_fallback",
            llm=llm,
            retrieval_pipeline=mock_retrieval,
        )
        cleanup.append(pipeline.pipeline_id)

        result = await pipeline._generate(query_id=1, top_k=2)

        assert result.text == "Fallback single answer"
        assert result.metadata is not None
        assert result.metadata["complexity_tier"] == "moderate"
        assert result.metadata["route"] == "single"
        mock_retrieval._retrieve_by_id.assert_awaited_once_with(1, 2)
        assert mock_retrieval.retrieve.await_count == 0

    @pytest.mark.asyncio
    async def test_multi_route_uses_configured_stop_query_signal_in_prompt(self, mock_retrieval):
        class StubService:
            def get_query_text(self, query_id: int | str) -> str:
                return f"query {query_id}"

            def get_chunk_contents(self, chunk_ids: list[int | str]) -> list[str]:
                return [f"chunk {chunk_id}" for chunk_id in chunk_ids]

        llm = FakeListLLM(responses=["complex", "DONE", "Complex answer"])
        mock_retrieval._retrieve_by_id = AsyncMock(return_value=[{"doc_id": 1, "score": 0.8}])
        mock_retrieval.retrieve = AsyncMock(return_value=[{"doc_id": 4, "score": 0.95}])
        pipeline = AdaptiveRAGPipeline.__new__(AdaptiveRAGPipeline)
        pipeline._service = StubService()
        pipeline._llm = llm
        pipeline._retrieval_pipeline = mock_retrieval
        pipeline._complexity_prompt_template = "{query}"
        pipeline._zero_retrieval_prompt_template = "{query}"
        pipeline._single_retrieval_prompt_template = "{context} {query}"
        pipeline._multi_retrieval_query_prompt_template = (
            "Question: {query}\nContext: {context}\nPrevious: {follow_up_queries}\n"
            "Respond with {stop_query_signal} when enough evidence is available."
        )
        pipeline._multi_retrieval_answer_prompt_template = "{query} {context} {follow_up_queries}"
        pipeline._route_for_simple = "zero"
        pipeline._route_for_moderate = "single"
        pipeline._route_for_complex = "multi"
        pipeline._max_multi_steps = 2
        pipeline._stop_query_signal = "DONE"

        result = await pipeline._generate(query_id=1, top_k=4)

        assert result.text == "Complex answer"
        assert result.metadata is not None
        assert result.metadata["follow_up_queries"] == []
        mock_retrieval._retrieve_by_id.assert_awaited_once_with(1, 4)
        assert mock_retrieval.retrieve.await_count == 0

    def test_config_rejects_invalid_route_values(self, mock_retrieval):
        llm = FakeListLLM(responses=["answer"])
        invalid_route = cast(Any, "singel")
        config = AdaptiveRAGPipelineConfig(
            name="adaptive_rag_invalid_route_cfg",
            retrieval_pipeline_name="bm25",
            llm=llm,
            route_for_simple=invalid_route,
        )
        config.inject_retrieval_pipeline(mock_retrieval)

        with pytest.raises(ValueError, match="route_for_simple"):
            config.get_pipeline_kwargs()

    def test_adaptive_rag_config(self, mock_retrieval):
        llm = FakeListLLM(responses=["answer"])
        config = AdaptiveRAGPipelineConfig(
            name="adaptive_rag_cfg",
            retrieval_pipeline_name="bm25",
            llm=llm,
            route_for_simple="single",
            route_for_moderate="multi",
            route_for_complex="zero",
            max_multi_steps=4,
            stop_query_signal="DONE",
        )
        config.inject_retrieval_pipeline(mock_retrieval)

        kwargs = config.get_pipeline_kwargs()

        assert config.get_pipeline_class() is AdaptiveRAGPipeline
        assert kwargs["llm"] is llm
        assert kwargs["retrieval_pipeline"] is mock_retrieval
        assert kwargs["route_for_simple"] == "single"
        assert kwargs["route_for_moderate"] == "multi"
        assert kwargs["route_for_complex"] == "zero"
        assert kwargs["max_multi_steps"] == 4
        assert kwargs["stop_query_signal"] == "DONE"

    def test_pipeline_config_output(self, session_factory, mock_retrieval, cleanup):
        llm = FakeListLLM(responses=["simple", "answer"])
        pipeline = AdaptiveRAGPipeline(
            session_factory=session_factory,
            name="test_adaptive_rag_config_output",
            llm=llm,
            retrieval_pipeline=mock_retrieval,
            route_for_simple="zero",
            route_for_moderate="single",
            route_for_complex="multi",
            max_multi_steps=3,
            stop_query_signal="HALT",
        )
        cleanup.append(pipeline.pipeline_id)

        config = pipeline._get_pipeline_config()

        assert config["type"] == "adaptive_rag"
        assert config["retrieval_pipeline_id"] == mock_retrieval.pipeline_id
        assert config["route_for_simple"] == "zero"
        assert config["route_for_moderate"] == "single"
        assert config["route_for_complex"] == "multi"
        assert config["max_multi_steps"] == 3
        assert config["stop_query_signal"] == "HALT"
        assert "complexity_prompt_template" in config
        assert "zero_retrieval_prompt_template" in config
        assert "single_retrieval_prompt_template" in config
        assert "multi_retrieval_query_prompt_template" in config
        assert "multi_retrieval_answer_prompt_template" in config

    def test_run_pipeline(self, session_factory, cleanup):
        from autorag_research.orm.repository.query import QueryRepository

        with session_factory() as session:
            query_repo = QueryRepository(session)
            query_count = query_repo.count()

        retrieval = create_mock_retrieval_pipeline()
        llm = create_mock_llm()

        pipeline = AdaptiveRAGPipeline(
            session_factory=session_factory,
            name="test_adaptive_rag_run",
            llm=llm,
            retrieval_pipeline=retrieval,
        )
        cleanup.append(pipeline.pipeline_id)

        result = pipeline.run(top_k=2, batch_size=10)

        config = PipelineTestConfig(
            pipeline_type="generation",
            expected_total_queries=query_count,
            check_token_usage=True,
            check_execution_time=True,
            check_persistence=True,
        )
        assert isinstance(pipeline.pipeline_id, int)
        verifier = PipelineTestVerifier(result, pipeline.pipeline_id, session_factory, config)
        verifier.verify_all()
