"""Tests for the prompt-based Self-RAG generation pipeline."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.autorag_research.pipelines.pipeline_test_utils import (
    PipelineTestConfig,
    PipelineTestVerifier,
    cleanup_pipeline_results_factory,
    create_mock_llm,
)


def create_sequential_llm(responses: list[str]) -> MagicMock:
    """Create a mock LLM that returns responses in sequence."""
    mock = MagicMock()
    call_count = [0]

    async def mock_ainvoke(prompt):
        response = MagicMock()
        response.content = responses[min(call_count[0], len(responses) - 1)]
        response.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        response.response_metadata = {}
        call_count[0] += 1
        return response

    mock.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    return mock


@pytest.fixture
def cleanup(session_factory):
    """Cleanup fixture for created pipeline results."""
    yield from cleanup_pipeline_results_factory(session_factory)


@pytest.fixture
def mock_retrieval_pipeline():
    """Create a retrieval pipeline mock for Self-RAG tests."""
    mock = MagicMock()
    mock.pipeline_id = 1
    mock.retrieve = AsyncMock(
        return_value=[
            {"doc_id": 1, "score": 0.95},
            {"doc_id": 2, "score": 0.85},
        ]
    )
    mock._retrieve_by_id = AsyncMock(return_value=[])
    return mock


class TestSelfRAGPipelineConfig:
    def test_config_exposes_pipeline_class_and_kwargs(self, mock_retrieval_pipeline):
        """SelfRAG config should return the pipeline class and injected kwargs."""
        from autorag_research.pipelines.generation.self_rag import SelfRAGPipeline, SelfRAGPipelineConfig

        config = SelfRAGPipelineConfig(
            name="self_rag_config",
            llm=create_mock_llm(),
            retrieval_pipeline_name="bm25_baseline",
            max_reflection_steps=3,
        )
        config.inject_retrieval_pipeline(mock_retrieval_pipeline)

        assert config.get_pipeline_class() is SelfRAGPipeline
        kwargs = config.get_pipeline_kwargs()
        assert kwargs["retrieval_pipeline"] is mock_retrieval_pipeline
        assert kwargs["max_reflection_steps"] == 3


class TestSelfRAGPipelineBehavior:
    def test_parse_reflection_treats_string_booleans_as_booleans(self):
        """Self-RAG JSON reflection parsing should honor quoted true/false values."""
        from autorag_research.pipelines.generation.self_rag import SelfRAGPipeline

        pipeline = SelfRAGPipeline.__new__(SelfRAGPipeline)

        reflection = pipeline._parse_reflection(
            '{"should_retrieve": "false", "is_supported": "false", "critique": "Need revision"}'
        )

        assert reflection == {
            "action": "REVISE",
            "supported": False,
            "search_query": "",
            "critique": "Need revision",
        }

    @pytest.mark.asyncio
    async def test_generate_skips_retrieval_when_initial_answer_is_supported(
        self, session_factory, cleanup, mock_retrieval_pipeline
    ):
        """Self-RAG should return the draft answer when reflection says no retrieval is needed."""
        from autorag_research.pipelines.generation.self_rag import SelfRAGPipeline

        pipeline = SelfRAGPipeline(
            session_factory=session_factory,
            name="test_self_rag_no_retrieval",
            llm=create_sequential_llm([
                "Draft answer without retrieval.",
                '{"should_retrieve": false, "is_supported": true, "critique": "Enough evidence in parametric memory."}',
            ]),
            retrieval_pipeline=mock_retrieval_pipeline,
        )
        cleanup.append(pipeline.pipeline_id)

        result = await pipeline._generate(query_id=1, top_k=2)

        assert result.text == "Draft answer without retrieval."
        assert result.metadata["used_retrieval"] is False
        assert result.metadata["support_passed"] is True
        assert result.metadata["retrieved_chunk_ids"] == []
        mock_retrieval_pipeline.retrieve.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_retrieves_and_revises_until_supported(
        self, session_factory, cleanup, mock_retrieval_pipeline
    ):
        """Self-RAG should retrieve evidence, revise, and stop once support passes."""
        from autorag_research.pipelines.generation.self_rag import SelfRAGPipeline

        pipeline = SelfRAGPipeline(
            session_factory=session_factory,
            name="test_self_rag_revision",
            llm=create_sequential_llm([
                "Initial uncertain answer.",
                '{"should_retrieve": true, "is_supported": false, "follow_up_query": "largest planet in solar system", "critique": "Need evidence."}',
                "Jupiter is the largest planet in the Solar System.",
                '{"should_retrieve": false, "is_supported": true, "critique": "The revised answer is supported."}',
            ]),
            retrieval_pipeline=mock_retrieval_pipeline,
            max_reflection_steps=2,
        )
        cleanup.append(pipeline.pipeline_id)

        result = await pipeline._generate(query_id=1, top_k=2)

        assert result.text == "Jupiter is the largest planet in the Solar System."
        assert result.metadata["used_retrieval"] is True
        assert result.metadata["support_passed"] is True
        assert result.metadata["reflection_iterations"] == 1
        assert result.metadata["retrieved_chunk_ids"] == [1, 2]
        mock_retrieval_pipeline.retrieve.assert_awaited_once_with("largest planet in solar system", 2)

    @pytest.mark.asyncio
    async def test_generate_stops_after_max_reflection_steps(self, session_factory, cleanup, mock_retrieval_pipeline):
        """Self-RAG should return the last revision when support never passes."""
        from autorag_research.pipelines.generation.self_rag import SelfRAGPipeline

        pipeline = SelfRAGPipeline(
            session_factory=session_factory,
            name="test_self_rag_max_steps",
            llm=create_sequential_llm([
                "Draft answer.",
                '{"should_retrieve": true, "is_supported": false, "follow_up_query": "largest planet", "critique": "Need evidence."}',
                "First revision.",
                '{"should_retrieve": true, "is_supported": false, "follow_up_query": "gas giant biggest", "critique": "Still unsupported."}',
                "Second revision.",
                '{"should_retrieve": true, "is_supported": false, "critique": "Still unsupported."}',
            ]),
            retrieval_pipeline=mock_retrieval_pipeline,
            max_reflection_steps=2,
        )
        cleanup.append(pipeline.pipeline_id)

        result = await pipeline._generate(query_id=1, top_k=2)

        assert result.text == "Second revision."
        assert result.metadata["used_retrieval"] is True
        assert result.metadata["support_passed"] is False
        assert result.metadata["reflection_iterations"] == 2
        assert mock_retrieval_pipeline.retrieve.await_count == 2

    def test_run_pipeline_with_verifier(self, session_factory, cleanup, mock_retrieval_pipeline):
        """Self-RAG should integrate with the standard generation pipeline runner."""
        from autorag_research.pipelines.generation.self_rag import SelfRAGPipeline

        llm = create_sequential_llm(
            [
                "Initial uncertain answer.",
                '{"should_retrieve": true, "is_supported": false, "follow_up_query": "largest planet in solar system", "critique": "Need evidence."}',
                "Jupiter is the largest planet in the Solar System.",
                '{"should_retrieve": false, "is_supported": true, "critique": "The revised answer is supported."}',
            ]
            * 5
        )

        pipeline = SelfRAGPipeline(
            session_factory=session_factory,
            name="test_self_rag_run",
            llm=llm,
            retrieval_pipeline=mock_retrieval_pipeline,
            max_reflection_steps=2,
        )
        cleanup.append(pipeline.pipeline_id)

        result = pipeline.run(top_k=2, batch_size=10)

        verifier = PipelineTestVerifier(
            result,
            pipeline.pipeline_id,
            session_factory,
            PipelineTestConfig(
                pipeline_type="generation",
                expected_total_queries=5,
                check_token_usage=True,
                check_execution_time=True,
                check_persistence=True,
            ),
        )
        report = verifier.verify_all()

        assert report.all_passed
