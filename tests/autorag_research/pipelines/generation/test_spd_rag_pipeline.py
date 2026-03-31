"""Tests for SPD-RAG (Sub-Agent Per Document) generation pipeline."""

from unittest.mock import MagicMock

import pytest

from autorag_research.orm.repository.executor_result import ExecutorResultRepository
from tests.autorag_research.pipelines.pipeline_test_utils import (
    PipelineTestConfig,
    PipelineTestVerifier,
    create_mock_llm,
)


class TestSPDRAGPipelineUnit:
    """Unit tests for SPD-RAG pipeline construction and config."""

    @pytest.fixture
    def mock_retrieval_pipeline(self):
        """Create a mock retrieval pipeline returning seed data chunk IDs."""
        mock = MagicMock()
        mock.pipeline_id = 1

        async def mock_retrieve(query_id: int | str, top_k: int):
            return [
                {"doc_id": 1, "score": 0.95},
                {"doc_id": 2, "score": 0.85},
                {"doc_id": 3, "score": 0.75},
                {"doc_id": 4, "score": 0.65},
                {"doc_id": 5, "score": 0.55},
            ][:top_k]

        mock._retrieve_by_id = mock_retrieve
        return mock

    @pytest.fixture
    def cleanup_pipeline_results(self, session_factory):
        """Cleanup fixture that deletes pipeline results after test."""
        created_pipeline_ids = []

        yield created_pipeline_ids

        session = session_factory()
        try:
            executor_repo = ExecutorResultRepository(session)
            for pipeline_id in created_pipeline_ids:
                executor_repo.delete_by_pipeline(pipeline_id)
            session.commit()
        finally:
            session.close()

    def test_pipeline_config_contains_required_fields(
        self, session_factory, mock_retrieval_pipeline, cleanup_pipeline_results
    ):
        """Test that pipeline config contains SPD-RAG specific fields."""
        from autorag_research.pipelines.generation.spd_rag import SPDRAGPipeline

        pipeline = SPDRAGPipeline(
            session_factory=session_factory,
            name="test_spd_rag_config",
            llm=create_mock_llm(),
            retrieval_pipeline=mock_retrieval_pipeline,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        config = pipeline._get_pipeline_config()

        assert config["type"] == "spd_rag"
        assert "sub_agent_system_prompt" in config
        assert "sub_agent_user_prompt" in config
        assert "coordinator_system_prompt" in config
        assert "coordinator_user_prompt" in config
        assert "synthesis_system_prompt" in config
        assert "synthesis_user_prompt" in config
        assert config["max_synthesis_tokens"] == 4000
        assert config["synthesis_batch_size"] == 3

    def test_pipeline_creation_with_default_parameters(
        self, session_factory, mock_retrieval_pipeline, cleanup_pipeline_results
    ):
        """Test pipeline creation with default parameters."""
        from autorag_research.pipelines.generation.spd_rag import SPDRAGPipeline

        pipeline = SPDRAGPipeline(
            session_factory=session_factory,
            name="test_spd_rag_defaults",
            llm=create_mock_llm(),
            retrieval_pipeline=mock_retrieval_pipeline,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        assert pipeline.pipeline_id > 0
        assert pipeline._max_synthesis_tokens == 4000
        assert pipeline._synthesis_batch_size == 3

    def test_pipeline_creation_with_custom_prompts(
        self, session_factory, mock_retrieval_pipeline, cleanup_pipeline_results
    ):
        """Test pipeline stores custom prompt templates."""
        from autorag_research.pipelines.generation.spd_rag import SPDRAGPipeline

        pipeline = SPDRAGPipeline(
            session_factory=session_factory,
            name="test_spd_rag_custom_prompts",
            llm=create_mock_llm(),
            retrieval_pipeline=mock_retrieval_pipeline,
            sub_agent_system_prompt="sub system",
            sub_agent_user_prompt="sub user {document} {query}",
            coordinator_system_prompt="coord system",
            coordinator_user_prompt="coord user {document} {partial_answer} {query}",
            synthesis_system_prompt="synth system",
            synthesis_user_prompt="synth user {partial_answers} {query}",
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        assert pipeline._sub_agent_system_prompt == "sub system"
        assert pipeline._sub_agent_user_prompt == "sub user {document} {query}"
        assert pipeline._coordinator_system_prompt == "coord system"
        assert pipeline._coordinator_user_prompt == "coord user {document} {partial_answer} {query}"
        assert pipeline._synthesis_system_prompt == "synth system"
        assert pipeline._synthesis_user_prompt == "synth user {partial_answers} {query}"


class TestSPDRAGPipelineIntegration:
    """Integration tests for SPD-RAG pipeline."""

    @pytest.fixture
    def mock_retrieval_pipeline(self):
        """Create a mock retrieval pipeline returning seed data chunk IDs."""
        mock = MagicMock()
        mock.pipeline_id = 1

        async def mock_retrieve(query_id: int | str, top_k: int):
            return [
                {"doc_id": 1, "score": 0.95},
                {"doc_id": 2, "score": 0.85},
                {"doc_id": 3, "score": 0.75},
            ][:top_k]

        mock._retrieve_by_id = mock_retrieve
        return mock

    @pytest.fixture
    def cleanup_pipeline_results(self, session_factory):
        """Cleanup fixture that deletes pipeline results after test."""
        created_pipeline_ids = []

        yield created_pipeline_ids

        session = session_factory()
        try:
            executor_repo = ExecutorResultRepository(session)
            for pipeline_id in created_pipeline_ids:
                executor_repo.delete_by_pipeline(pipeline_id)
            session.commit()
        finally:
            session.close()

    @staticmethod
    def _create_mock_llm_with_multi_call_responses(
        num_docs: int = 3,
        synthesis_batch_size: int = 3,
        token_usage: dict[str, int] | None = None,
    ) -> MagicMock:
        """Create a mock LLM that cycles through sub-agent, coordinator, and synthesis calls."""
        if token_usage is None:
            token_usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

        mock = MagicMock()
        call_count = [0]
        synthesis_calls = 0 if num_docs <= 1 else (num_docs + synthesis_batch_size - 1) // synthesis_batch_size
        calls_per_query = (2 * num_docs) + synthesis_calls

        def create_response(idx: int) -> MagicMock:
            response = MagicMock()
            phase_idx = idx % calls_per_query

            if phase_idx < num_docs:
                response.content = f"Partial answer {phase_idx + 1}"
            elif phase_idx < 2 * num_docs:
                response.content = "Yes"
            else:
                response.content = "Synthesized final answer."

            response.response_metadata = {}
            response.usage_metadata = token_usage
            return response

        async def mock_ainvoke(messages):
            idx = call_count[0]
            call_count[0] += 1
            return create_response(idx)

        mock.ainvoke = mock_ainvoke
        return mock

    def test_run_pipeline_with_verifier(self, session_factory, mock_retrieval_pipeline, cleanup_pipeline_results):
        """Test full SPD-RAG pipeline run with standard verifier checks."""
        from autorag_research.pipelines.generation.spd_rag import SPDRAGPipeline

        pipeline = SPDRAGPipeline(
            session_factory=session_factory,
            name="test_spd_rag_integration",
            llm=self._create_mock_llm_with_multi_call_responses(),
            retrieval_pipeline=mock_retrieval_pipeline,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        result = pipeline.run(top_k=3, batch_size=10)

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

    @pytest.mark.asyncio
    async def test_token_usage_aggregates_all_phases(
        self, session_factory, mock_retrieval_pipeline, cleanup_pipeline_results
    ):
        """Test token usage aggregates sub-agent, coordinator, and synthesis calls."""
        from autorag_research.pipelines.generation.spd_rag import SPDRAGPipeline

        pipeline = SPDRAGPipeline(
            session_factory=session_factory,
            name="test_spd_rag_token_usage",
            llm=self._create_mock_llm_with_multi_call_responses(
                num_docs=3,
                token_usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            ),
            retrieval_pipeline=mock_retrieval_pipeline,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        result = await pipeline._generate(query_id=1, top_k=3)

        assert result.token_usage["prompt_tokens"] == 70
        assert result.token_usage["completion_tokens"] == 35
        assert result.token_usage["total_tokens"] == 105


class TestSPDRAGEdgeCases:
    """Edge case tests for SPD-RAG pipeline behavior."""

    @pytest.fixture
    def cleanup_pipeline_results(self, session_factory):
        """Cleanup fixture that deletes pipeline results after test."""
        created_pipeline_ids = []

        yield created_pipeline_ids

        session = session_factory()
        try:
            executor_repo = ExecutorResultRepository(session)
            for pipeline_id in created_pipeline_ids:
                executor_repo.delete_by_pipeline(pipeline_id)
            session.commit()
        finally:
            session.close()

    @pytest.mark.asyncio
    async def test_empty_retrieval_returns_error_metadata(self, session_factory, cleanup_pipeline_results):
        """Test empty retrieval returns error metadata and empty text."""
        from autorag_research.pipelines.generation.spd_rag import SPDRAGPipeline

        mock_retrieval = MagicMock()
        mock_retrieval.pipeline_id = 1

        async def mock_retrieve(query_id: int | str, top_k: int):
            return []

        mock_retrieval._retrieve_by_id = mock_retrieve

        pipeline = SPDRAGPipeline(
            session_factory=session_factory,
            name="test_spd_rag_empty_retrieval",
            llm=create_mock_llm(),
            retrieval_pipeline=mock_retrieval,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        result = await pipeline._generate(query_id=1, top_k=3)

        assert result.text == ""
        assert result.metadata["error"] == "No documents retrieved"
        assert result.metadata["pipeline_type"] == "spd_rag"
        assert result.token_usage["total_tokens"] == 0

    @pytest.mark.asyncio
    async def test_single_document_skips_synthesis(self, session_factory, cleanup_pipeline_results):
        """Test a single relevant answer is returned directly without synthesis."""
        from autorag_research.pipelines.generation.spd_rag import SPDRAGPipeline

        mock_retrieval = MagicMock()
        mock_retrieval.pipeline_id = 1

        async def mock_retrieve(query_id: int | str, top_k: int):
            return [{"doc_id": 1, "score": 0.95}]

        mock_retrieval._retrieve_by_id = mock_retrieve

        mock_llm = MagicMock()
        call_count = [0]

        def create_response(idx: int) -> MagicMock:
            response = MagicMock()
            response.content = "Single document partial answer" if idx == 0 else "Yes"
            response.response_metadata = {}
            response.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
            return response

        async def mock_ainvoke(messages):
            idx = call_count[0]
            call_count[0] += 1
            return create_response(idx)

        mock_llm.ainvoke = mock_ainvoke

        pipeline = SPDRAGPipeline(
            session_factory=session_factory,
            name="test_spd_rag_single_doc",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        result = await pipeline._generate(query_id=1, top_k=1)

        assert result.text == "Single document partial answer"
        assert result.metadata["relevant_answer_count"] == 1
        assert call_count[0] == 2

    @pytest.mark.asyncio
    async def test_coordinator_filters_all_uses_fallback(self, session_factory, cleanup_pipeline_results):
        """Test coordinator fallback uses all partial answers when none are marked relevant."""
        from autorag_research.pipelines.generation.spd_rag import SPDRAGPipeline

        mock_retrieval = MagicMock()
        mock_retrieval.pipeline_id = 1

        async def mock_retrieve(query_id: int | str, top_k: int):
            return [
                {"doc_id": 1, "score": 0.95},
                {"doc_id": 2, "score": 0.85},
            ]

        mock_retrieval._retrieve_by_id = mock_retrieve

        mock_llm = MagicMock()
        call_count = [0]

        def create_response(idx: int) -> MagicMock:
            response = MagicMock()
            if idx < 2:
                response.content = f"Partial answer {idx + 1}"
            elif idx < 4:
                response.content = "No"
            else:
                response.content = "Fallback synthesized answer"
            response.response_metadata = {}
            response.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
            return response

        async def mock_ainvoke(messages):
            idx = call_count[0]
            call_count[0] += 1
            return create_response(idx)

        mock_llm.ainvoke = mock_ainvoke

        pipeline = SPDRAGPipeline(
            session_factory=session_factory,
            name="test_spd_rag_fallback",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        result = await pipeline._generate(query_id=1, top_k=2)

        assert result.text == "Fallback synthesized answer"
        assert result.metadata["used_coordinator_fallback"] is True
        assert result.metadata["relevant_answer_count"] == 2

    @pytest.mark.asyncio
    async def test_metadata_contains_pipeline_statistics(self, session_factory, cleanup_pipeline_results):
        """Test metadata includes SPD-RAG pipeline statistics."""
        from autorag_research.pipelines.generation.spd_rag import SPDRAGPipeline

        mock_retrieval = MagicMock()
        mock_retrieval.pipeline_id = 1

        async def mock_retrieve(query_id: int | str, top_k: int):
            return [
                {"doc_id": 1, "score": 0.95},
                {"doc_id": 2, "score": 0.85},
                {"doc_id": 3, "score": 0.75},
            ]

        mock_retrieval._retrieve_by_id = mock_retrieve

        mock_llm = MagicMock()
        call_count = [0]

        def create_response(idx: int) -> MagicMock:
            response = MagicMock()
            if idx < 3:
                response.content = f"Partial answer {idx + 1}"
            elif idx < 6:
                response.content = "Yes"
            else:
                response.content = "Synthesized answer"
            response.response_metadata = {}
            response.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
            return response

        async def mock_ainvoke(messages):
            idx = call_count[0]
            call_count[0] += 1
            return create_response(idx)

        mock_llm.ainvoke = mock_ainvoke

        pipeline = SPDRAGPipeline(
            session_factory=session_factory,
            name="test_spd_rag_metadata",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        result = await pipeline._generate(query_id=1, top_k=3)

        assert result.metadata["pipeline_type"] == "spd_rag"
        assert result.metadata["original_doc_count"] == 3
        assert result.metadata["relevant_answer_count"] == 3
        assert result.metadata["retrieved_chunk_ids"] == [1, 2, 3]
        assert result.metadata["retrieval_scores"] == [0.95, 0.85, 0.75]
        assert result.metadata["synthesis_batch_size"] == 3
        assert result.metadata["max_synthesis_tokens"] == 4000


class TestSPDRAGPipelineConfig:
    """Tests for SPDRAGPipelineConfig dataclass."""

    def test_config_get_pipeline_class(self):
        """Test config returns the correct pipeline class."""
        from autorag_research.pipelines.generation.spd_rag import SPDRAGPipeline, SPDRAGPipelineConfig

        config = SPDRAGPipelineConfig(
            name="test_config",
            retrieval_pipeline_name="test_retrieval",
            llm=create_mock_llm(),
        )

        assert config.get_pipeline_class() == SPDRAGPipeline

    def test_config_get_pipeline_kwargs_without_injection_raises(self):
        """Test get_pipeline_kwargs raises when retrieval pipeline is not injected."""
        from autorag_research.pipelines.generation.spd_rag import SPDRAGPipelineConfig

        config = SPDRAGPipelineConfig(
            name="test_config",
            retrieval_pipeline_name="test_retrieval",
            llm=create_mock_llm(),
        )

        with pytest.raises(ValueError, match="not injected"):
            config.get_pipeline_kwargs()

    def test_config_get_pipeline_kwargs_with_injection(self):
        """Test get_pipeline_kwargs returns SPD-RAG constructor args."""
        from autorag_research.pipelines.generation.spd_rag import SPDRAGPipelineConfig

        mock_llm = create_mock_llm()
        mock_retrieval = MagicMock()

        config = SPDRAGPipelineConfig(
            name="test_config",
            retrieval_pipeline_name="test_retrieval",
            llm=mock_llm,
            max_synthesis_tokens=2048,
            synthesis_batch_size=4,
        )
        config._retrieval_pipeline = mock_retrieval

        kwargs = config.get_pipeline_kwargs()

        assert kwargs["llm"] == mock_llm
        assert kwargs["retrieval_pipeline"] == mock_retrieval
        assert kwargs["max_synthesis_tokens"] == 2048
        assert kwargs["synthesis_batch_size"] == 4
        assert "sub_agent_system_prompt" in kwargs
        assert "sub_agent_user_prompt" in kwargs
        assert "coordinator_system_prompt" in kwargs
        assert "coordinator_user_prompt" in kwargs
        assert "synthesis_system_prompt" in kwargs
        assert "synthesis_user_prompt" in kwargs

    def test_config_default_synthesis_parameters(self):
        """Test default synthesis parameters match the design."""
        from autorag_research.pipelines.generation.spd_rag import SPDRAGPipelineConfig

        config = SPDRAGPipelineConfig(
            name="test_config",
            retrieval_pipeline_name="test_retrieval",
            llm=create_mock_llm(),
        )

        assert config.synthesis_batch_size == 3
        assert config.max_synthesis_tokens == 4000
