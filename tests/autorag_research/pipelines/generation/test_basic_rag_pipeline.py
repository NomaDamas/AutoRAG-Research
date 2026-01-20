import pytest

from autorag_research.orm.repository.chunk_retrieved_result import ChunkRetrievedResultRepository
from autorag_research.orm.repository.executor_result import ExecutorResultRepository
from autorag_research.pipelines.generation.basic_rag import BasicRAGPipeline
from autorag_research.pipelines.retrieval.bm25 import BM25RetrievalPipeline
from tests.autorag_research.pipelines.pipeline_test_utils import (
    PipelineTestConfig,
    PipelineTestVerifier,
    create_mock_llm,
)


class TestBasicRAGPipeline:
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM that returns predictable responses.

        Note: We mock LLM because API calls are expensive and require keys.
        The retrieval pipeline uses real BM25 index from seed data.
        """
        return create_mock_llm(
            response_text="This is a generated answer.",
            token_usage={
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        )

    @pytest.fixture
    def retrieval_pipeline(self, session_factory, bm25_index_path, cleanup_pipeline_results):
        """Create a real BM25RetrievalPipeline with index from seed data."""
        pipeline = BM25RetrievalPipeline(
            session_factory=session_factory,
            name="test_retrieval_for_rag",
            index_path=bm25_index_path,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)
        return pipeline

    @pytest.fixture
    def cleanup_pipeline_results(self, session_factory):
        """Cleanup fixture that deletes pipeline results after test."""
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
    def pipeline(self, session_factory, mock_llm, retrieval_pipeline, cleanup_pipeline_results):
        """Create a BasicRAGPipeline with real retrieval and mock LLM."""
        pipeline = BasicRAGPipeline(
            session_factory=session_factory,
            name="test_basic_rag_pipeline",
            llm=mock_llm,
            retrieval_pipeline=retrieval_pipeline,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)
        return pipeline

    def test_pipeline_creation(self, pipeline, mock_llm, retrieval_pipeline):
        """Test that pipeline is created correctly."""
        assert pipeline.pipeline_id > 0
        assert pipeline._llm == mock_llm
        assert pipeline._retrieval_pipeline == retrieval_pipeline

    def test_pipeline_config(self, pipeline, retrieval_pipeline):
        """Test that pipeline config is stored correctly."""
        config = pipeline._get_pipeline_config()
        assert config["type"] == "basic_rag"
        assert config["retrieval_pipeline_id"] == retrieval_pipeline.pipeline_id
        assert "prompt_template" in config

    def test_generate_single_query(self, pipeline, mock_llm):
        """Test generation for a single query."""
        # Use query that matches seed data chunks
        result = pipeline._generate("What is Chunk about?", top_k=3)

        assert result.text == "This is a generated answer."
        assert result.token_usage is not None
        assert result.token_usage["total_tokens"] == 150
        assert result.metadata is not None
        assert "retrieved_chunk_ids" in result.metadata

        # Verify LLM was called
        mock_llm.complete.assert_called_once()

    def test_run_pipeline(self, pipeline, mock_llm, session_factory):
        """Test running the full pipeline using the test framework."""
        result = pipeline.run(top_k=2, batch_size=10)

        config = PipelineTestConfig(
            pipeline_type="generation",
            expected_total_queries=5,  # Seed data has 5 queries
            check_token_usage=True,
            check_execution_time=True,
            check_persistence=True,
        )
        verifier = PipelineTestVerifier(result, pipeline.pipeline_id, session_factory, config)
        verifier.verify_all()

        # Additional specific checks for token aggregation
        assert result["token_usage"]["total_tokens"] == 750  # 5 queries * 150 tokens each
        assert result["token_usage"]["prompt_tokens"] == 500  # 5 queries * 100 tokens each
        assert result["token_usage"]["completion_tokens"] == 250  # 5 queries * 50 tokens each

    def test_custom_prompt_template(self, session_factory, mock_llm, retrieval_pipeline, cleanup_pipeline_results):
        """Test pipeline with custom prompt template."""
        custom_template = "Documents:\n{context}\n\nQuery: {query}\n\nResponse:"

        pipeline = BasicRAGPipeline(
            session_factory=session_factory,
            name="test_custom_template_pipeline",
            llm=mock_llm,
            retrieval_pipeline=retrieval_pipeline,
            prompt_template=custom_template,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        _ = pipeline._generate("Test query", top_k=2)

        # Verify the custom template was used by checking the call
        call_args = mock_llm.complete.call_args
        prompt = call_args[0][0]
        assert "Documents:" in prompt
        assert "Response:" in prompt

    def test_empty_retrieval_results(self, session_factory, mock_llm, bm25_index_path, cleanup_pipeline_results):
        """Test handling of empty retrieval results."""
        # Create retrieval pipeline
        retrieval = BM25RetrievalPipeline(
            session_factory=session_factory,
            name="test_empty_retrieval",
            index_path=bm25_index_path,
        )
        cleanup_pipeline_results.append(retrieval.pipeline_id)

        pipeline = BasicRAGPipeline(
            session_factory=session_factory,
            name="test_empty_retrieval_pipeline",
            llm=mock_llm,
            retrieval_pipeline=retrieval,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        # Query with no likely matches in seed data
        result = pipeline._generate("xyzzy nonexistent query", top_k=5)

        # Should still produce a result even with no/few context
        assert result.text is not None

    def test_token_counting_handler_initialization(self, pipeline):
        """Test that TokenCountingHandler is initialized properly."""
        assert hasattr(pipeline, "_token_counter")
        assert pipeline._token_counter is not None

        # Verify the callback manager is set on the LLM
        assert pipeline._llm.callback_manager is not None

    def test_token_usage_dict_structure(self, pipeline):
        """Test that _generate returns token_usage as a dict with expected keys."""
        result = pipeline._generate("What is AI?", top_k=2)

        assert result.token_usage is not None
        assert "prompt_tokens" in result.token_usage
        assert "completion_tokens" in result.token_usage
        assert "total_tokens" in result.token_usage
        assert "embedding_tokens" in result.token_usage

    def test_token_counter_reset_between_generations(self, pipeline):
        """Test that token counter is reset between generations."""
        # First generation
        result1 = pipeline._generate("First query", top_k=2)

        # Second generation
        result2 = pipeline._generate("Second query", top_k=2)

        # Token counts should be independent (not accumulated)
        assert result1.token_usage is not None
        assert result2.token_usage is not None
        assert "total_tokens" in result1.token_usage
        assert "total_tokens" in result2.token_usage

    def test_token_usage_total_consistency(self, pipeline):
        """Test that total_tokens equals prompt_tokens + completion_tokens."""
        result = pipeline._generate("Test query", top_k=2)

        if result.token_usage:
            expected_total = result.token_usage["prompt_tokens"] + result.token_usage["completion_tokens"]
            assert result.token_usage["total_tokens"] == expected_total
