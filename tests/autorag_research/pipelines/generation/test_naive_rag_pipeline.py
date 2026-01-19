from unittest.mock import MagicMock

import pytest

from autorag_research.orm.repository.chunk_retrieved_result import ChunkRetrievedResultRepository
from autorag_research.orm.repository.executor_result import ExecutorResultRepository
from autorag_research.pipelines.generation.naive_rag import NaiveRAGPipeline


class TestNaiveRAGPipeline:
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM that returns predictable responses."""
        mock = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "This is a generated answer."
        mock_response.__str__ = lambda x: "This is a generated answer."
        mock_response.raw = {"usage": {"total_tokens": 150}}
        mock.complete.return_value = mock_response
        return mock

    @pytest.fixture
    def mock_retrieval_pipeline(self, session_factory):
        """Create a mock retrieval pipeline."""
        mock = MagicMock()
        mock.pipeline_id = 1

        def mock_retrieve(query_text: str, top_k: int):
            # Return mock chunk IDs that exist in seed data
            return [
                {"doc_id": 1, "score": 0.9},
                {"doc_id": 2, "score": 0.8},
                {"doc_id": 3, "score": 0.7},
            ][:top_k]

        mock.retrieve = mock_retrieve
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
        pipeline = NaiveRAGPipeline(
            session_factory=session_factory,
            name="test_naive_rag_pipeline",
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
        assert config["type"] == "naive_rag"
        assert config["retrieval_pipeline_id"] == mock_retrieval_pipeline.pipeline_id
        assert "prompt_template" in config

    def test_generate_single_query(self, pipeline, mock_llm, mock_retrieval_pipeline):
        """Test generation for a single query."""
        result = pipeline._generate("What is the meaning of life?", top_k=3)

        assert result.text == "This is a generated answer."
        assert result.token_usage == 150
        assert result.metadata is not None
        assert "retrieved_chunk_ids" in result.metadata
        assert len(result.metadata["retrieved_chunk_ids"]) == 3

        # Verify LLM was called
        mock_llm.complete.assert_called_once()

    def test_run_pipeline(self, pipeline, mock_llm):
        """Test running the full pipeline."""
        result = pipeline.run(top_k=2, batch_size=10)

        assert "pipeline_id" in result
        assert "total_queries" in result
        assert "total_tokens" in result
        assert "avg_execution_time_ms" in result
        assert result["pipeline_id"] == pipeline.pipeline_id
        assert result["total_queries"] == 5  # Seed data has 5 queries
        assert result["total_tokens"] == 750  # 5 queries * 150 tokens each

    def test_custom_prompt_template(self, session_factory, mock_llm, mock_retrieval_pipeline, cleanup_pipeline_results):
        """Test pipeline with custom prompt template."""
        custom_template = "Documents:\n{context}\n\nQuery: {query}\n\nResponse:"

        pipeline = NaiveRAGPipeline(
            session_factory=session_factory,
            name="test_custom_template_pipeline",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
            prompt_template=custom_template,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        _ = pipeline._generate("Test query", top_k=2)

        # Verify the custom template was used by checking the call
        call_args = mock_llm.complete.call_args
        prompt = call_args[0][0]
        assert "Documents:" in prompt
        assert "Response:" in prompt

    def test_empty_retrieval_results(self, session_factory, mock_llm, cleanup_pipeline_results):
        """Test handling of empty retrieval results."""
        mock_retrieval = MagicMock()
        mock_retrieval.pipeline_id = 999
        mock_retrieval.retrieve.return_value = []

        pipeline = NaiveRAGPipeline(
            session_factory=session_factory,
            name="test_empty_retrieval_pipeline",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        result = pipeline._generate("Query with no results", top_k=5)

        # Should still produce a result even with no context
        assert result.text is not None
        assert result.metadata["retrieved_chunk_ids"] == []
