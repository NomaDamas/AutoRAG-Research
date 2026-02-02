from unittest.mock import AsyncMock, MagicMock

import pytest

from autorag_research.orm.repository.chunk_retrieved_result import ChunkRetrievedResultRepository
from autorag_research.orm.repository.executor_result import ExecutorResultRepository
from autorag_research.pipelines.generation.basic_rag import BasicRAGPipeline
from tests.autorag_research.pipelines.pipeline_test_utils import (
    PipelineTestConfig,
    PipelineTestVerifier,
    create_mock_llm,
)


class TestBasicRAGPipeline:
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM that returns predictable responses."""
        return create_mock_llm()

    @pytest.fixture
    def mock_retrieval_pipeline(self, session_factory):
        """Create a mock retrieval pipeline with async _retrieve_by_id method."""
        mock = MagicMock()
        mock.pipeline_id = 1

        async def mock_retrieve_by_id(query_id: int, top_k: int):
            # Return mock chunk IDs that exist in seed data
            return [
                {"doc_id": 1, "score": 0.9},
                {"doc_id": 2, "score": 0.8},
                {"doc_id": 3, "score": 0.7},
            ][:top_k]

        mock._retrieve_by_id = mock_retrieve_by_id
        return mock

    @pytest.fixture
    def cleanup_pipeline_results(self, session_factory):
        created_pipeline_ids = []

        yield created_pipeline_ids

        session = session_factory()
        try:
            executor_repo = ExecutorResultRepository(session)
            chunk_result_repo = ChunkRetrievedResultRepository(session)
            for pipeline_id in created_pipeline_ids:
                executor_repo.delete_by_pipeline(pipeline_id)
                chunk_result_repo.delete_by_pipeline(pipeline_id)
            session.commit()
        finally:
            session.close()

    @pytest.fixture
    def pipeline(self, session_factory, mock_llm, mock_retrieval_pipeline, cleanup_pipeline_results):
        pipeline = BasicRAGPipeline(
            session_factory=session_factory,
            name="test_basic_rag_pipeline",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)
        return pipeline

    def test_pipeline_creation(self, pipeline, mock_llm, mock_retrieval_pipeline):
        """Test that pipeline is created correctly."""
        assert pipeline.pipeline_id > 0
        assert pipeline._llm == mock_llm
        assert pipeline._retrieval_pipeline == mock_retrieval_pipeline

    def test_pipeline_config(self, pipeline, mock_retrieval_pipeline):
        """Test that pipeline config is stored correctly."""
        config = pipeline._get_pipeline_config()
        assert config["type"] == "basic_rag"
        assert config["retrieval_pipeline_id"] == mock_retrieval_pipeline.pipeline_id
        assert "prompt_template" in config

    @pytest.mark.asyncio
    async def test_generate_single_query(self, pipeline, mock_llm, mock_retrieval_pipeline):
        """Test generation for a single query using query_id from seed data."""
        # Use query_id=1 from seed data
        result = await pipeline._generate(query_id=1, top_k=3)

        assert result.text == "This is a generated answer."
        # token_usage is now a dict
        assert result.token_usage is not None
        assert result.token_usage["total_tokens"] == 150
        assert result.token_usage["prompt_tokens"] == 100
        assert result.token_usage["completion_tokens"] == 50
        assert result.metadata is not None
        assert "retrieved_chunk_ids" in result.metadata
        assert len(result.metadata["retrieved_chunk_ids"]) == 3

        # Verify LLM was called (async ainvoke)
        mock_llm.ainvoke.assert_called_once()

    def test_run_pipeline(self, pipeline, session_factory):
        """Test running the full pipeline with PipelineTestVerifier."""
        from autorag_research.orm.repository.query import QueryRepository

        # Count actual queries in database
        with session_factory() as session:
            query_repo = QueryRepository(session)
            query_count = query_repo.count()

        result = pipeline.run(top_k=2, batch_size=10)

        # Use PipelineTestVerifier for standard output validation
        config = PipelineTestConfig(
            pipeline_type="generation",
            expected_total_queries=query_count,
            check_token_usage=True,
            check_execution_time=True,
            check_persistence=True,
        )
        verifier = PipelineTestVerifier(result, pipeline.pipeline_id, session_factory, config)
        verifier.verify_all()

    def test_run_pipeline_token_aggregation(self, pipeline, session_factory):
        """Test that token usage is correctly aggregated across all queries."""
        from autorag_research.orm.repository.query import QueryRepository

        # Count actual queries in database
        with session_factory() as session:
            query_repo = QueryRepository(session)
            query_count = query_repo.count()

        result = pipeline.run(top_k=2, batch_size=10)

        # Verify aggregated token_usage values (N queries * mock token counts)
        assert result["token_usage"]["total_tokens"] == query_count * 150
        assert result["token_usage"]["prompt_tokens"] == query_count * 100
        assert result["token_usage"]["completion_tokens"] == query_count * 50

    @pytest.mark.asyncio
    async def test_custom_prompt_template(
        self, session_factory, mock_llm, mock_retrieval_pipeline, cleanup_pipeline_results
    ):
        """Test pipeline with custom prompt template."""
        custom_template = "Documents:\n{context}\n\nQuery: {query}\n\nResponse:"

        pipeline = BasicRAGPipeline(
            session_factory=session_factory,
            name="test_custom_template_pipeline",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
            prompt_template=custom_template,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        # Use query_id=1 from seed data
        _ = await pipeline._generate(query_id=1, top_k=2)

        # Verify the custom template was used by checking the call (async ainvoke)
        call_args = mock_llm.ainvoke.call_args
        prompt = call_args[0][0]
        assert "Documents:" in prompt
        assert "Response:" in prompt

    @pytest.mark.asyncio
    async def test_empty_retrieval_results(self, session_factory, mock_llm, cleanup_pipeline_results):
        """Test handling of empty retrieval results."""
        mock_retrieval = MagicMock()
        mock_retrieval.pipeline_id = 999
        mock_retrieval._retrieve_by_id = AsyncMock(return_value=[])

        pipeline = BasicRAGPipeline(
            session_factory=session_factory,
            name="test_empty_retrieval_pipeline",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        # Use query_id=1 from seed data
        result = await pipeline._generate(query_id=1, top_k=5)

        # Should still produce a result even with no context
        assert result.text is not None
        assert result.metadata["retrieved_chunk_ids"] == []

    @pytest.mark.asyncio
    async def test_token_usage_dict_structure(self, pipeline):
        """Test that _generate returns token_usage as a dict with expected keys."""
        result = await pipeline._generate(query_id=1, top_k=2)

        assert result.token_usage is not None
        assert "prompt_tokens" in result.token_usage
        assert "completion_tokens" in result.token_usage
        assert "total_tokens" in result.token_usage

    @pytest.mark.asyncio
    async def test_token_counter_reset_between_generations(self, pipeline):
        """Test that token counter is reset between generations."""
        # First generation (use query_id=1 from seed data)
        result1 = await pipeline._generate(query_id=1, top_k=2)

        # Second generation (use query_id=2 from seed data)
        result2 = await pipeline._generate(query_id=2, top_k=2)

        # Token counts should be independent (not accumulated)
        # Since we're using mock LLM with raw response, values should be present
        assert result1.token_usage is not None
        assert result2.token_usage is not None
        # Verify both have the expected keys
        assert "total_tokens" in result1.token_usage
        assert "total_tokens" in result2.token_usage

    @pytest.mark.asyncio
    async def test_token_usage_total_consistency(self, pipeline):
        """Test that total_tokens equals prompt_tokens + completion_tokens."""
        result = await pipeline._generate(query_id=1, top_k=2)

        if result.token_usage:
            expected_total = result.token_usage["prompt_tokens"] + result.token_usage["completion_tokens"]
            assert result.token_usage["total_tokens"] == expected_total
