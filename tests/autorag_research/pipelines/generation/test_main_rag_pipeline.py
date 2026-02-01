"""Tests for MAIN-RAG (Multi-Agent Filtering RAG) Pipeline.

MAIN-RAG uses three LLM agents to collaboratively filter retrieved documents
through adaptive thresholding and probabilistic scoring:
- Agent-1 (Predictor): Generates candidate answers per document
- Agent-2 (Judge): Scores document relevance via log probabilities
- Agent-3 (Final Predictor): Generates final answer from filtered docs

Test Strategy:
- Unit tests for adaptive threshold calculation (MAIN-RAG specific logic)
- Integration tests using PipelineTestVerifier for standard validation
- Edge case tests for filtering scenarios
"""

from statistics import mean, stdev
from unittest.mock import MagicMock

import pytest

from autorag_research.orm.repository.executor_result import ExecutorResultRepository
from tests.autorag_research.pipelines.pipeline_test_utils import (
    PipelineTestConfig,
    PipelineTestVerifier,
    create_mock_llm,
)


class TestMAINRAGPipelineUnit:
    """Unit tests for MAINRAGPipeline inner logic.

    These tests focus on MAIN-RAG specific algorithms that are NOT covered
    by PipelineTestVerifier:
    - Adaptive threshold calculation
    - Pipeline configuration storage
    - Multi-agent coordination
    - LogprobsNotSupportedError handling
    """

    @pytest.fixture
    def mock_retrieval_pipeline(self):
        """Create a mock retrieval pipeline."""
        mock = MagicMock()
        mock.pipeline_id = 1

        def mock_retrieve(query_text: str, top_k: int):
            # Return mock chunk IDs that simulate retrieval results
            return [
                {"doc_id": 1, "score": 0.95},
                {"doc_id": 2, "score": 0.85},
                {"doc_id": 3, "score": 0.75},
                {"doc_id": 4, "score": 0.65},
                {"doc_id": 5, "score": 0.55},
            ][:top_k]

        mock.retrieve = mock_retrieve
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

    # ==================== Adaptive Threshold Tests ====================

    def test_calculate_adaptive_threshold_normal_case(self):
        """Test threshold calculation with varied scores.

        Formula: threshold = mean - n * std
        With n=0 (default), threshold = mean
        """
        from autorag_research.pipelines.generation.main_rag import MAINRAGPipeline

        scores = [5.0, 3.0, 1.0, -1.0, -3.0]
        expected_mean = mean(scores)  # 1.0
        expected_std = stdev(scores)  # ~3.16

        # Test with std_multiplier = 0 (default)
        threshold = MAINRAGPipeline._calculate_adaptive_threshold(scores, std_multiplier=0.0)
        assert abs(threshold - expected_mean) < 0.001

        # Test with std_multiplier = 1
        threshold = MAINRAGPipeline._calculate_adaptive_threshold(scores, std_multiplier=1.0)
        expected = expected_mean - 1.0 * expected_std
        assert abs(threshold - expected) < 0.001

    def test_calculate_adaptive_threshold_all_same_scores(self):
        """Test threshold when all scores are identical (zero std deviation).

        When std = 0, threshold should equal the mean regardless of std_multiplier.
        """
        from autorag_research.pipelines.generation.main_rag import MAINRAGPipeline

        scores = [2.5, 2.5, 2.5, 2.5]

        threshold = MAINRAGPipeline._calculate_adaptive_threshold(scores, std_multiplier=1.0)

        assert threshold == 2.5

    def test_calculate_adaptive_threshold_single_score(self):
        """Test threshold with single score returns that score."""
        from autorag_research.pipelines.generation.main_rag import MAINRAGPipeline

        scores = [3.14]

        threshold = MAINRAGPipeline._calculate_adaptive_threshold(scores, std_multiplier=0.0)

        assert threshold == 3.14

    def test_calculate_adaptive_threshold_empty_scores_raises_error(self):
        """Test that empty scores list raises ValueError."""
        from autorag_research.pipelines.generation.main_rag import MAINRAGPipeline

        with pytest.raises(ValueError, match="empty"):
            MAINRAGPipeline._calculate_adaptive_threshold([], std_multiplier=0.0)

    def test_calculate_adaptive_threshold_negative_multiplier(self):
        """Test threshold with negative std_multiplier (more aggressive filtering).

        Negative multiplier increases threshold: threshold = mean - (-n) * std = mean + n * std
        """
        from autorag_research.pipelines.generation.main_rag import MAINRAGPipeline

        scores = [5.0, 3.0, 1.0, -1.0, -3.0]
        expected_mean = mean(scores)  # 1.0
        expected_std = stdev(scores)  # ~3.16

        # Negative multiplier should increase threshold (more aggressive filtering)
        threshold = MAINRAGPipeline._calculate_adaptive_threshold(scores, std_multiplier=-1.0)
        expected = expected_mean - (-1.0) * expected_std  # mean + std

        assert abs(threshold - expected) < 0.001
        assert threshold > expected_mean  # Higher threshold means more aggressive filtering

    def test_calculate_adaptive_threshold_high_multiplier(self):
        """Test threshold with high std_multiplier (permissive filtering).

        Higher multiplier lowers threshold: more documents pass
        """
        from autorag_research.pipelines.generation.main_rag import MAINRAGPipeline

        scores = [5.0, 3.0, 1.0, -1.0, -3.0]
        expected_mean = mean(scores)  # 1.0
        expected_std = stdev(scores)  # ~3.16

        threshold = MAINRAGPipeline._calculate_adaptive_threshold(scores, std_multiplier=2.0)
        expected = expected_mean - 2.0 * expected_std

        assert abs(threshold - expected) < 0.001
        assert threshold < expected_mean  # Lower threshold is more permissive

    # ==================== LogprobsNotSupportedError Tests ====================

    def test_agent_judge_raises_error_without_logprobs(
        self, session_factory, mock_retrieval_pipeline, cleanup_pipeline_results
    ):
        """Test that _agent_judge raises LogprobsNotSupportedError when LLM doesn't support logprobs.

        MAIN-RAG requires logprobs for accurate scoring. If the LLM doesn't provide them,
        the pipeline should fail-fast with a clear error message.
        """
        from autorag_research.exceptions import LogprobsNotSupportedError
        from autorag_research.pipelines.generation.main_rag import MAINRAGPipeline

        # Create mock LLM WITHOUT logprobs
        mock_llm = create_mock_llm(response_text="Yes", include_logprobs=False)

        pipeline = MAINRAGPipeline(
            session_factory=session_factory,
            name="test_main_rag_no_logprobs",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        with pytest.raises(LogprobsNotSupportedError) as exc_info:
            pipeline._agent_judge("test query", "test document", "test answer")

        assert "test_main_rag_no_logprobs" in str(exc_info.value)
        assert "logprobs" in str(exc_info.value).lower()

    def test_generate_raises_error_without_logprobs(
        self, session_factory, mock_retrieval_pipeline, cleanup_pipeline_results
    ):
        """Test that _generate raises LogprobsNotSupportedError when LLM doesn't support logprobs.

        The error should be raised during the Agent-2 (Judge) phase when multiple documents
        are retrieved and filtering is needed.
        """
        from autorag_research.exceptions import LogprobsNotSupportedError
        from autorag_research.pipelines.generation.main_rag import MAINRAGPipeline

        # Create mock LLM WITHOUT logprobs
        mock_llm = create_mock_llm(response_text="Yes", include_logprobs=False)

        pipeline = MAINRAGPipeline(
            session_factory=session_factory,
            name="test_main_rag_generate_no_logprobs",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        # This should raise LogprobsNotSupportedError during the Judge phase
        with pytest.raises(LogprobsNotSupportedError):
            pipeline._generate("test query", top_k=3)

    # ==================== Pipeline Configuration Tests ====================

    def test_pipeline_config_contains_required_fields(
        self, session_factory, mock_retrieval_pipeline, cleanup_pipeline_results
    ):
        """Test that pipeline config contains MAIN-RAG specific fields."""
        from autorag_research.pipelines.generation.main_rag import MAINRAGPipeline

        mock_llm = create_mock_llm()

        pipeline = MAINRAGPipeline(
            session_factory=session_factory,
            name="test_main_rag_config",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
            std_multiplier=0.5,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        config = pipeline._get_pipeline_config()

        assert config["type"] == "main_rag"
        assert config["std_multiplier"] == 0.5
        assert "predictor_system_prompt" in config
        assert "predictor_user_prompt" in config
        assert "judge_system_prompt" in config
        assert "judge_user_prompt" in config
        assert "final_predictor_system_prompt" in config
        assert "final_predictor_user_prompt" in config

    def test_pipeline_creation_with_default_parameters(
        self, session_factory, mock_retrieval_pipeline, cleanup_pipeline_results
    ):
        """Test pipeline creation with default parameters."""
        from autorag_research.pipelines.generation.main_rag import MAINRAGPipeline

        mock_llm = create_mock_llm()

        pipeline = MAINRAGPipeline(
            session_factory=session_factory,
            name="test_main_rag_defaults",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        assert pipeline.pipeline_id > 0
        assert pipeline._std_multiplier == 0.0  # Default from design doc

    def test_pipeline_creation_with_custom_prompts(
        self, session_factory, mock_retrieval_pipeline, cleanup_pipeline_results
    ):
        """Test pipeline creation with custom prompt templates."""
        from autorag_research.pipelines.generation.main_rag import MAINRAGPipeline

        mock_llm = create_mock_llm()
        custom_predictor_system = "Custom predictor system instructions"
        custom_predictor_user = "Custom predictor: {document} {query}"
        custom_judge_system = "Custom judge system instructions"
        custom_judge_user = "Custom judge: {document} {query} {answer}"
        custom_final_system = "Custom final system instructions"
        custom_final_user = "Custom final: {documents} {query}"

        pipeline = MAINRAGPipeline(
            session_factory=session_factory,
            name="test_main_rag_custom_prompts",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
            predictor_system_prompt=custom_predictor_system,
            predictor_user_prompt=custom_predictor_user,
            judge_system_prompt=custom_judge_system,
            judge_user_prompt=custom_judge_user,
            final_predictor_system_prompt=custom_final_system,
            final_predictor_user_prompt=custom_final_user,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        config = pipeline._get_pipeline_config()
        assert config["predictor_system_prompt"] == custom_predictor_system
        assert config["predictor_user_prompt"] == custom_predictor_user
        assert config["judge_system_prompt"] == custom_judge_system
        assert config["judge_user_prompt"] == custom_judge_user
        assert config["final_predictor_system_prompt"] == custom_final_system
        assert config["final_predictor_user_prompt"] == custom_final_user


class TestMAINRAGPipelineIntegration:
    """Integration tests for MAINRAGPipeline.

    Uses PipelineTestVerifier for standard output validation.
    Does NOT duplicate what verify_all() already checks.
    """

    @pytest.fixture
    def mock_retrieval_pipeline(self):
        """Create a mock retrieval pipeline returning seed data chunk IDs."""
        mock = MagicMock()
        mock.pipeline_id = 1

        def mock_retrieve(query_text: str, top_k: int):
            # Return chunk IDs that exist in seed data (002-seed.sql)
            return [
                {"doc_id": 1, "score": 0.95},
                {"doc_id": 2, "score": 0.85},
                {"doc_id": 3, "score": 0.75},
            ][:top_k]

        mock.retrieve = mock_retrieve
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

    def _create_mock_llm_with_multi_call_responses(self, num_docs: int = 3) -> MagicMock:
        """Create mock LLM that returns different responses for multi-agent flow.

        MAIN-RAG makes multiple LLM calls per query:
        - num_docs calls to Agent-1 (Predictor) - one per document
        - num_docs calls to Agent-2 (Judge) - one per document
        - 1 call to Agent-3 (Final Predictor)

        Total: 2*num_docs + 1 calls per query

        Supports both sync invoke() and async ainvoke() methods.
        """
        mock = MagicMock()
        call_count = [0]  # Use list to allow mutation in closure

        def create_response(idx: int) -> MagicMock:
            response = MagicMock()

            # Determine which agent based on call index
            calls_per_query = 2 * num_docs + 1

            agent_idx = idx % calls_per_query
            if agent_idx < num_docs:
                # Agent-1: Predictor - return candidate answer
                response.content = f"Candidate answer {agent_idx + 1}"
                response.response_metadata = {}
            elif agent_idx < 2 * num_docs:
                # Agent-2: Judge - return Yes/No with logprobs
                response.content = "Yes"
                response.response_metadata = {
                    "logprobs": {
                        "content": [
                            {
                                "token": "Yes",
                                "logprob": -0.1,
                                "bytes": [89, 101, 115],
                                "top_logprobs": [
                                    {"token": "Yes", "logprob": -0.1, "bytes": [89, 101, 115]},
                                    {"token": "No", "logprob": -2.5, "bytes": [78, 111]},
                                ],
                            }
                        ]
                    }
                }
            else:
                # Agent-3: Final Predictor - return final answer
                response.content = "This is the final generated answer based on filtered documents."
                response.response_metadata = {}

            # Set token usage for all responses
            response.usage_metadata = {
                "input_tokens": 50,
                "output_tokens": 20,
                "total_tokens": 70,
            }

            return response

        def mock_invoke(prompt):
            idx = call_count[0]
            call_count[0] += 1
            return create_response(idx)

        async def mock_ainvoke(prompt):
            idx = call_count[0]
            call_count[0] += 1
            return create_response(idx)

        mock.invoke = mock_invoke
        mock.ainvoke = mock_ainvoke
        return mock

    def test_run_pipeline_with_verifier(self, session_factory, mock_retrieval_pipeline, cleanup_pipeline_results):
        """Test full pipeline execution with PipelineTestVerifier.

        PipelineTestVerifier handles:
        - Return structure validation
        - pipeline_id matching
        - total_queries count
        - token_usage structure
        - avg_execution_time_ms validity
        - Database persistence

        This test only sets up the pipeline; verifier does the heavy lifting.
        """
        from autorag_research.pipelines.generation.main_rag import MAINRAGPipeline

        mock_llm = self._create_mock_llm_with_multi_call_responses(num_docs=3)

        pipeline = MAINRAGPipeline(
            session_factory=session_factory,
            name="test_main_rag_verifier",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        result = pipeline.run(top_k=3, batch_size=10)

        # Use PipelineTestVerifier for standard validation
        config = PipelineTestConfig(
            pipeline_type="generation",
            expected_total_queries=5,  # Seed data has 5 queries (002-seed.sql)
            check_token_usage=True,
            check_execution_time=True,
            check_persistence=True,
        )
        verifier = PipelineTestVerifier(result, pipeline.pipeline_id, session_factory, config)
        verifier.verify_all()

    def test_token_usage_aggregates_all_agents(
        self, session_factory, mock_retrieval_pipeline, cleanup_pipeline_results
    ):
        """Test that token usage is correctly aggregated across all three agents.

        MAIN-RAG specific: verify tokens from Predictor, Judge, and Final Predictor
        are all counted.
        """
        from autorag_research.pipelines.generation.main_rag import MAINRAGPipeline

        num_docs = 3
        mock_llm = self._create_mock_llm_with_multi_call_responses(num_docs=num_docs)

        pipeline = MAINRAGPipeline(
            session_factory=session_factory,
            name="test_main_rag_token_aggregation",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        result = pipeline.run(top_k=num_docs, batch_size=10)

        # Verify token aggregation
        # Per query: 2*num_docs + 1 calls = 7 calls with 70 tokens each = 490 tokens per query
        # 5 queries (002-seed.sql) * 490 = 2450 total tokens
        expected_calls_per_query = 2 * num_docs + 1  # 7
        expected_tokens_per_call = 70
        expected_total_tokens = 5 * expected_calls_per_query * expected_tokens_per_call

        assert result["token_usage"] is not None
        assert result["token_usage"]["total_tokens"] == expected_total_tokens


class TestMAINRAGEdgeCases:
    """Test edge cases specific to MAIN-RAG multi-agent filtering.

    These tests verify MAIN-RAG specific behavior not covered by PipelineTestVerifier.
    """

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

    def test_empty_retrieval_returns_error_metadata(self, session_factory, cleanup_pipeline_results):
        """Test that empty retrieval results produce error in metadata."""
        from autorag_research.pipelines.generation.main_rag import MAINRAGPipeline

        mock_llm = create_mock_llm()
        mock_retrieval = MagicMock()
        mock_retrieval.pipeline_id = 999
        mock_retrieval.retrieve.return_value = []

        pipeline = MAINRAGPipeline(
            session_factory=session_factory,
            name="test_main_rag_empty_retrieval",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        result = pipeline._generate("Query with no results", top_k=5)

        # Should handle gracefully with error metadata
        assert result.text == ""
        assert result.metadata is not None
        assert "error" in result.metadata

    def test_all_documents_filtered_uses_top_one(self, session_factory, cleanup_pipeline_results):
        """Test that when all documents are filtered, the top-scoring document is used.

        MAIN-RAG design: never produce empty results - use best available doc.
        """
        from autorag_research.pipelines.generation.main_rag import MAINRAGPipeline

        # Create mock LLM that gives all documents negative scores (all "No" judgments)
        mock_llm = MagicMock()
        call_count = [0]

        def create_response(idx: int) -> MagicMock:
            response = MagicMock()

            if idx < 2:
                # Predictor calls
                response.content = f"Answer {idx}"
                response.response_metadata = {}
            elif idx < 4:
                # Judge calls - all "No" (documents not relevant)
                response.content = "No"
                response.response_metadata = {
                    "logprobs": {
                        "content": [
                            {
                                "token": "No",
                                "logprob": -0.01,
                                "bytes": [78, 111],
                                "top_logprobs": [
                                    {"token": "No", "logprob": -0.01, "bytes": [78, 111]},
                                    {"token": "Yes", "logprob": -5.0, "bytes": [89, 101, 115]},
                                ],
                            }
                        ]
                    }
                }
            else:
                # Final predictor
                response.content = "Answer using fallback document."
                response.response_metadata = {}

            response.usage_metadata = {"input_tokens": 50, "output_tokens": 20, "total_tokens": 70}
            return response

        def mock_invoke(prompt):
            idx = call_count[0]
            call_count[0] += 1
            return create_response(idx)

        async def mock_ainvoke(prompt):
            idx = call_count[0]
            call_count[0] += 1
            return create_response(idx)

        mock_llm.invoke = mock_invoke
        mock_llm.ainvoke = mock_ainvoke

        mock_retrieval = MagicMock()
        mock_retrieval.pipeline_id = 1
        mock_retrieval.retrieve.return_value = [
            {"doc_id": 1, "score": 0.9},
            {"doc_id": 2, "score": 0.7},
        ]

        pipeline = MAINRAGPipeline(
            session_factory=session_factory,
            name="test_main_rag_all_filtered",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        result = pipeline._generate("Test query", top_k=2)

        # Should still produce an answer using the top-scoring document
        assert result.text != ""
        assert result.metadata is not None
        assert result.metadata.get("filtered_doc_count", 0) >= 1

    def test_single_document_skips_filtering(self, session_factory, cleanup_pipeline_results):
        """Test that single document retrieval skips the filtering phase.

        With only one document, filtering would be meaningless.
        """
        from autorag_research.pipelines.generation.main_rag import MAINRAGPipeline

        mock_llm = MagicMock()
        call_count = [0]

        def create_response(idx: int) -> MagicMock:
            response = MagicMock()

            if idx == 0:
                # Only one predictor call
                response.content = "Single doc answer"
            else:
                # Final predictor (skip judge for single doc)
                response.content = "Final answer from single document."

            response.response_metadata = {}
            response.usage_metadata = {"input_tokens": 50, "output_tokens": 20, "total_tokens": 70}
            return response

        def mock_invoke(prompt):
            idx = call_count[0]
            call_count[0] += 1
            return create_response(idx)

        async def mock_ainvoke(prompt):
            idx = call_count[0]
            call_count[0] += 1
            return create_response(idx)

        mock_llm.invoke = mock_invoke
        mock_llm.ainvoke = mock_ainvoke

        mock_retrieval = MagicMock()
        mock_retrieval.pipeline_id = 1
        mock_retrieval.retrieve.return_value = [{"doc_id": 1, "score": 0.95}]

        pipeline = MAINRAGPipeline(
            session_factory=session_factory,
            name="test_main_rag_single_doc",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        result = pipeline._generate("Test query", top_k=1)

        # Should produce result and indicate filtering was skipped
        assert result.text != ""
        assert result.metadata is not None
        assert result.metadata.get("skipped_filtering") is True

    def test_filtering_preserves_document_order_by_score(self, session_factory, cleanup_pipeline_results):
        """Test that filtered documents are ordered by relevance score (descending)."""
        from autorag_research.pipelines.generation.main_rag import MAINRAGPipeline

        mock_llm = MagicMock()
        call_count = [0]
        judge_scores = [-0.5, 2.0, 1.0]  # Doc 2 highest, then doc 3, then doc 1

        def create_response(idx: int) -> MagicMock:
            response = MagicMock()

            if idx < 3:
                response.content = f"Answer for doc {idx + 1}"
                response.response_metadata = {}
            elif idx < 6:
                # Judge calls
                doc_idx = idx - 3
                score = judge_scores[doc_idx]
                is_yes = score > 0
                response.content = "Yes" if is_yes else "No"
                response.response_metadata = {
                    "logprobs": {
                        "content": [
                            {
                                "token": "Yes" if is_yes else "No",
                                "logprob": -0.1 if is_yes else -2.0,
                                "bytes": [89, 101, 115] if is_yes else [78, 111],
                                "top_logprobs": [
                                    {"token": "Yes", "logprob": -0.1 if is_yes else -2.0, "bytes": [89, 101, 115]},
                                    {"token": "No", "logprob": -2.0 + score if is_yes else -0.1, "bytes": [78, 111]},
                                ],
                            }
                        ]
                    }
                }
            else:
                response.content = "Final answer."
                response.response_metadata = {}

            response.usage_metadata = {"input_tokens": 50, "output_tokens": 20, "total_tokens": 70}
            return response

        def mock_invoke(prompt):
            idx = call_count[0]
            call_count[0] += 1
            return create_response(idx)

        async def mock_ainvoke(prompt):
            idx = call_count[0]
            call_count[0] += 1
            return create_response(idx)

        mock_llm.invoke = mock_invoke
        mock_llm.ainvoke = mock_ainvoke

        mock_retrieval = MagicMock()
        mock_retrieval.pipeline_id = 1
        mock_retrieval.retrieve.return_value = [
            {"doc_id": 1, "score": 0.95},
            {"doc_id": 2, "score": 0.85},
            {"doc_id": 3, "score": 0.75},
        ]

        pipeline = MAINRAGPipeline(
            session_factory=session_factory,
            name="test_main_rag_order",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        result = pipeline._generate("Test query", top_k=3)

        # Check that relevance_scores in metadata are sorted descending
        assert result.metadata is not None
        relevance_scores = result.metadata.get("relevance_scores", [])
        if len(relevance_scores) > 1:
            scores = [s["score"] for s in relevance_scores]
            assert scores == sorted(scores, reverse=True), "Scores should be in descending order"

    def test_metadata_contains_filtering_statistics(self, session_factory, cleanup_pipeline_results):
        """Test that result metadata contains filtering statistics."""
        from autorag_research.pipelines.generation.main_rag import MAINRAGPipeline

        mock_llm = MagicMock()
        call_count = [0]

        def create_response(idx: int) -> MagicMock:
            response = MagicMock()

            if idx < 3:
                response.content = f"Answer {idx}"
                response.response_metadata = {}
            elif idx < 6:
                response.content = "Yes"
                response.response_metadata = {
                    "logprobs": {
                        "content": [
                            {
                                "token": "Yes",
                                "logprob": -0.1,
                                "bytes": [89, 101, 115],
                                "top_logprobs": [
                                    {"token": "Yes", "logprob": -0.1, "bytes": [89, 101, 115]},
                                    {"token": "No", "logprob": -2.5, "bytes": [78, 111]},
                                ],
                            }
                        ]
                    }
                }
            else:
                response.content = "Final answer."
                response.response_metadata = {}

            response.usage_metadata = {"input_tokens": 50, "output_tokens": 20, "total_tokens": 70}
            return response

        def mock_invoke(prompt):
            idx = call_count[0]
            call_count[0] += 1
            return create_response(idx)

        async def mock_ainvoke(prompt):
            idx = call_count[0]
            call_count[0] += 1
            return create_response(idx)

        mock_llm.invoke = mock_invoke
        mock_llm.ainvoke = mock_ainvoke

        mock_retrieval = MagicMock()
        mock_retrieval.pipeline_id = 1
        mock_retrieval.retrieve.return_value = [
            {"doc_id": 1, "score": 0.95},
            {"doc_id": 2, "score": 0.85},
            {"doc_id": 3, "score": 0.75},
        ]

        pipeline = MAINRAGPipeline(
            session_factory=session_factory,
            name="test_main_rag_metadata",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        result = pipeline._generate("Test query", top_k=3)

        # Verify metadata contains MAIN-RAG specific statistics
        assert result.metadata is not None
        assert "pipeline_type" in result.metadata
        assert result.metadata["pipeline_type"] == "main_rag"
        assert "original_doc_count" in result.metadata
        assert "filtered_doc_count" in result.metadata
        assert "threshold" in result.metadata
        assert "std_multiplier" in result.metadata


class TestMAINRAGParallelExecution:
    """Tests for MAIN-RAG parallel execution using asyncio.

    These tests verify that:
    1. Async agent methods work correctly
    2. Parallel execution produces same results as sequential
    3. Token usage is correctly aggregated from parallel calls
    4. Error handling works in parallel context
    """

    @pytest.fixture
    def mock_retrieval_pipeline(self):
        """Create a mock retrieval pipeline returning seed data chunk IDs."""
        mock = MagicMock()
        mock.pipeline_id = 1

        def mock_retrieve(query_text: str, top_k: int):
            # Return chunk IDs that exist in seed data (002-seed.sql)
            return [
                {"doc_id": 1, "score": 0.95},
                {"doc_id": 2, "score": 0.85},
                {"doc_id": 3, "score": 0.75},
            ][:top_k]

        mock.retrieve = mock_retrieve
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

    def _create_async_mock_llm(self, num_docs: int = 3) -> MagicMock:
        """Create mock LLM with both sync invoke and async ainvoke methods.

        This mock tracks call count and returns different responses for
        Predictor/Judge/Final Predictor phases, supporting both sync and async calls.
        """
        import asyncio

        mock = MagicMock()
        call_count = [0]  # Use list to allow mutation in closure

        def create_response(idx: int) -> MagicMock:
            response = MagicMock()
            # Determine which agent based on call index
            calls_per_query = 2 * num_docs + 1

            agent_idx = idx % calls_per_query
            if agent_idx < num_docs:
                # Agent-1: Predictor - return candidate answer
                response.content = f"Candidate answer {agent_idx + 1}"
                response.response_metadata = {}
            elif agent_idx < 2 * num_docs:
                # Agent-2: Judge - return Yes/No with logprobs
                response.content = "Yes"
                response.response_metadata = {
                    "logprobs": {
                        "content": [
                            {
                                "token": "Yes",
                                "logprob": -0.1,
                                "bytes": [89, 101, 115],
                                "top_logprobs": [
                                    {"token": "Yes", "logprob": -0.1, "bytes": [89, 101, 115]},
                                    {"token": "No", "logprob": -2.5, "bytes": [78, 111]},
                                ],
                            }
                        ]
                    }
                }
            else:
                # Agent-3: Final Predictor - return final answer
                response.content = "This is the final generated answer based on filtered documents."
                response.response_metadata = {}

            # Set token usage for all responses
            response.usage_metadata = {
                "input_tokens": 50,
                "output_tokens": 20,
                "total_tokens": 70,
            }

            return response

        def mock_invoke(prompt):
            idx = call_count[0]
            call_count[0] += 1
            return create_response(idx)

        async def mock_ainvoke(prompt):
            idx = call_count[0]
            call_count[0] += 1
            # Add small delay to simulate real async behavior
            await asyncio.sleep(0.001)
            return create_response(idx)

        mock.invoke = mock_invoke
        mock.ainvoke = mock_ainvoke
        return mock

    @pytest.mark.asyncio
    async def test_aagent_predict_returns_correct_result(
        self, session_factory, mock_retrieval_pipeline, cleanup_pipeline_results
    ):
        """Test that async _aagent_predict returns the same result as sync version."""
        from autorag_research.pipelines.generation.main_rag import MAINRAGPipeline

        mock_llm = self._create_async_mock_llm(num_docs=1)

        pipeline = MAINRAGPipeline(
            session_factory=session_factory,
            name="test_async_predict",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        # Call async predict
        answer, usage = await pipeline._aagent_predict("test query", "test document")

        assert answer == "Candidate answer 1"
        assert usage is not None
        assert usage["total_tokens"] == 70

    @pytest.mark.asyncio
    async def test_aagent_judge_returns_correct_score(
        self, session_factory, mock_retrieval_pipeline, cleanup_pipeline_results
    ):
        """Test that async _aagent_judge returns correct score from logprobs."""
        from autorag_research.pipelines.generation.main_rag import MAINRAGPipeline

        # Need to set up mock to return judge response first
        mock_llm = MagicMock()

        async def mock_ainvoke(prompt):
            response = MagicMock()
            response.content = "Yes"
            response.response_metadata = {
                "logprobs": {
                    "content": [
                        {
                            "token": "Yes",
                            "logprob": -0.1,
                            "bytes": [89, 101, 115],
                            "top_logprobs": [
                                {"token": "Yes", "logprob": -0.1, "bytes": [89, 101, 115]},
                                {"token": "No", "logprob": -2.5, "bytes": [78, 111]},
                            ],
                        }
                    ]
                }
            }
            response.usage_metadata = {
                "input_tokens": 50,
                "output_tokens": 20,
                "total_tokens": 70,
            }
            return response

        mock_llm.ainvoke = mock_ainvoke
        mock_llm.invoke = MagicMock()

        pipeline = MAINRAGPipeline(
            session_factory=session_factory,
            name="test_async_judge",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        # Call async judge
        score, usage = await pipeline._aagent_judge("test query", "test document", "test answer")

        # Score should be log P(Yes) - log P(No) = -0.1 - (-2.5) = 2.4
        assert abs(score - 2.4) < 0.001
        assert usage is not None
        assert usage["total_tokens"] == 70

    @pytest.mark.asyncio
    async def test_aagent_judge_raises_logprobs_not_supported(
        self, session_factory, mock_retrieval_pipeline, cleanup_pipeline_results
    ):
        """Test that async _aagent_judge raises LogprobsNotSupportedError when no logprobs."""
        from autorag_research.exceptions import LogprobsNotSupportedError
        from autorag_research.pipelines.generation.main_rag import MAINRAGPipeline

        mock_llm = MagicMock()

        async def mock_ainvoke(prompt):
            response = MagicMock()
            response.content = "Yes"
            response.response_metadata = {}  # No logprobs
            response.usage_metadata = {
                "input_tokens": 50,
                "output_tokens": 20,
                "total_tokens": 70,
            }
            return response

        mock_llm.ainvoke = mock_ainvoke
        mock_llm.invoke = MagicMock()

        pipeline = MAINRAGPipeline(
            session_factory=session_factory,
            name="test_async_judge_no_logprobs",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        with pytest.raises(LogprobsNotSupportedError):
            await pipeline._aagent_judge("test query", "test document", "test answer")

    @pytest.mark.asyncio
    async def test_run_predictors_parallel_executes_all_documents(
        self, session_factory, mock_retrieval_pipeline, cleanup_pipeline_results
    ):
        """Test that _run_predictors_parallel executes for all documents."""
        from autorag_research.pipelines.generation.main_rag import MAINRAGPipeline

        mock_llm = self._create_async_mock_llm(num_docs=3)

        pipeline = MAINRAGPipeline(
            session_factory=session_factory,
            name="test_parallel_predictors",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        documents = ["doc1", "doc2", "doc3"]
        answers, usages = await pipeline._run_predictors_parallel("test query", documents, max_concurrency=3)

        assert len(answers) == 3
        assert len(usages) == 3
        # All answers should be non-empty
        for answer in answers:
            assert answer != ""
        # All usages should have token counts
        for usage in usages:
            assert usage.get("total_tokens", 0) > 0

    @pytest.mark.asyncio
    async def test_run_judges_parallel_executes_all_documents(
        self, session_factory, mock_retrieval_pipeline, cleanup_pipeline_results
    ):
        """Test that _run_judges_parallel executes for all (document, answer) pairs."""
        from autorag_research.pipelines.generation.main_rag import MAINRAGPipeline

        # Create mock that always returns judge response with logprobs
        mock_llm = MagicMock()
        call_count = [0]

        async def mock_ainvoke(prompt):
            call_count[0] += 1
            response = MagicMock()
            response.content = "Yes"
            response.response_metadata = {
                "logprobs": {
                    "content": [
                        {
                            "token": "Yes",
                            "logprob": -0.1,
                            "bytes": [89, 101, 115],
                            "top_logprobs": [
                                {"token": "Yes", "logprob": -0.1, "bytes": [89, 101, 115]},
                                {"token": "No", "logprob": -2.5, "bytes": [78, 111]},
                            ],
                        }
                    ]
                }
            }
            response.usage_metadata = {
                "input_tokens": 50,
                "output_tokens": 20,
                "total_tokens": 70,
            }
            return response

        mock_llm.ainvoke = mock_ainvoke
        mock_llm.invoke = MagicMock()

        pipeline = MAINRAGPipeline(
            session_factory=session_factory,
            name="test_parallel_judges",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        documents = ["doc1", "doc2", "doc3"]
        answers = ["answer1", "answer2", "answer3"]
        scores, usages = await pipeline._run_judges_parallel("test query", documents, answers, max_concurrency=3)

        assert len(scores) == 3
        assert len(usages) == 3
        # All scores should be the same (2.4) since mock returns same logprobs
        for score in scores:
            assert abs(score - 2.4) < 0.001
        # All usages should have token counts
        for usage in usages:
            assert usage.get("total_tokens", 0) > 0

    def test_generate_with_parallel_execution(self, session_factory, mock_retrieval_pipeline, cleanup_pipeline_results):
        """Test that _generate with parallel execution produces correct results."""
        from autorag_research.pipelines.generation.main_rag import MAINRAGPipeline

        mock_llm = self._create_async_mock_llm(num_docs=3)

        pipeline = MAINRAGPipeline(
            session_factory=session_factory,
            name="test_generate_parallel",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        result = pipeline._generate("test query", top_k=3, max_concurrency=3)

        assert result.text != ""
        assert result.token_usage is not None
        # Should have token usage aggregated from all 7 calls (3 predictor + 3 judge + 1 final)
        expected_total_tokens = 7 * 70  # 490
        assert result.token_usage["total_tokens"] == expected_total_tokens

    def test_generate_default_max_concurrency(self, session_factory, mock_retrieval_pipeline, cleanup_pipeline_results):
        """Test that _generate has a sensible default max_concurrency."""
        from autorag_research.pipelines.generation.main_rag import MAINRAGPipeline

        mock_llm = self._create_async_mock_llm(num_docs=3)

        pipeline = MAINRAGPipeline(
            session_factory=session_factory,
            name="test_generate_default_concurrency",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        # Call without specifying max_concurrency
        result = pipeline._generate("test query", top_k=3)

        assert result.text != ""
        assert result.token_usage is not None

    def test_run_passes_batch_size_as_max_concurrency(
        self, session_factory, mock_retrieval_pipeline, cleanup_pipeline_results
    ):
        """Test that run() passes batch_size as max_concurrency to _generate."""
        from autorag_research.pipelines.generation.main_rag import MAINRAGPipeline

        mock_llm = self._create_async_mock_llm(num_docs=3)

        pipeline = MAINRAGPipeline(
            session_factory=session_factory,
            name="test_run_batch_size",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        # Run with specific batch_size
        result = pipeline.run(top_k=3, batch_size=5)

        # Verify pipeline executed (result structure checked by verifier in other tests)
        assert result["pipeline_id"] == pipeline.pipeline_id
        assert result["total_queries"] > 0

    def test_token_usage_aggregation_with_parallel_execution(
        self, session_factory, mock_retrieval_pipeline, cleanup_pipeline_results
    ):
        """Test that token usage is correctly aggregated from parallel calls."""
        from autorag_research.pipelines.generation.main_rag import MAINRAGPipeline

        mock_llm = self._create_async_mock_llm(num_docs=3)

        pipeline = MAINRAGPipeline(
            session_factory=session_factory,
            name="test_token_aggregation_parallel",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        result = pipeline._generate("test query", top_k=3, max_concurrency=3)

        # Per call: 50 prompt + 20 completion = 70 total
        # Total calls: 3 predictor + 3 judge + 1 final = 7
        assert result.token_usage is not None
        assert result.token_usage["prompt_tokens"] == 7 * 50  # 350
        assert result.token_usage["completion_tokens"] == 7 * 20  # 140
        assert result.token_usage["total_tokens"] == 7 * 70  # 490

    def test_single_document_uses_async_methods(self, session_factory, cleanup_pipeline_results):
        """Test that single document case uses async methods correctly."""
        from autorag_research.pipelines.generation.main_rag import MAINRAGPipeline

        mock_llm = MagicMock()
        call_count = [0]

        async def mock_ainvoke(prompt):
            idx = call_count[0]
            call_count[0] += 1
            response = MagicMock()
            if idx == 0:
                response.content = "Single doc answer"
            else:
                response.content = "Final answer from single document."
            response.response_metadata = {}
            response.usage_metadata = {
                "input_tokens": 50,
                "output_tokens": 20,
                "total_tokens": 70,
            }
            return response

        mock_llm.ainvoke = mock_ainvoke
        mock_llm.invoke = MagicMock()

        mock_retrieval = MagicMock()
        mock_retrieval.pipeline_id = 1
        mock_retrieval.retrieve.return_value = [{"doc_id": 1, "score": 0.95}]

        pipeline = MAINRAGPipeline(
            session_factory=session_factory,
            name="test_single_doc_async",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        result = pipeline._generate("test query", top_k=1)

        assert result.text != ""
        assert result.metadata.get("skipped_filtering") is True
        # Should have 2 calls (predictor + final)
        assert result.token_usage["total_tokens"] == 2 * 70


class TestMAINRAGPipelineConfig:
    """Tests for MAINRAGPipelineConfig dataclass."""

    def test_config_get_pipeline_class(self):
        """Test that config returns correct pipeline class."""
        from autorag_research.pipelines.generation.main_rag import MAINRAGPipeline, MAINRAGPipelineConfig

        config = MAINRAGPipelineConfig(
            name="test_config",
            retrieval_pipeline_name="test_retrieval",
            llm=create_mock_llm(),
        )

        assert config.get_pipeline_class() == MAINRAGPipeline

    def test_config_get_pipeline_kwargs_without_injection_raises(self):
        """Test that get_pipeline_kwargs raises error when retrieval pipeline not injected."""
        from autorag_research.pipelines.generation.main_rag import MAINRAGPipelineConfig

        config = MAINRAGPipelineConfig(
            name="test_config",
            retrieval_pipeline_name="test_retrieval",
            llm=create_mock_llm(),
        )

        with pytest.raises(ValueError, match="not injected"):
            config.get_pipeline_kwargs()

    def test_config_get_pipeline_kwargs_with_injection(self):
        """Test that get_pipeline_kwargs returns correct dict when retrieval pipeline is injected."""
        from autorag_research.pipelines.generation.main_rag import MAINRAGPipelineConfig

        mock_llm = create_mock_llm()
        mock_retrieval = MagicMock()

        config = MAINRAGPipelineConfig(
            name="test_config",
            retrieval_pipeline_name="test_retrieval",
            llm=mock_llm,
            std_multiplier=0.5,
        )
        config._retrieval_pipeline = mock_retrieval

        kwargs = config.get_pipeline_kwargs()

        assert kwargs["llm"] == mock_llm
        assert kwargs["retrieval_pipeline"] == mock_retrieval
        assert kwargs["std_multiplier"] == 0.5
        assert "predictor_system_prompt" in kwargs
        assert "predictor_user_prompt" in kwargs
        assert "judge_system_prompt" in kwargs
        assert "judge_user_prompt" in kwargs
        assert "final_predictor_system_prompt" in kwargs
        assert "final_predictor_user_prompt" in kwargs

    def test_config_default_std_multiplier(self):
        """Test that default std_multiplier is 0.0 per paper recommendation."""
        from autorag_research.pipelines.generation.main_rag import MAINRAGPipelineConfig

        config = MAINRAGPipelineConfig(
            name="test_config",
            retrieval_pipeline_name="test_retrieval",
            llm=create_mock_llm(),
        )

        assert config.std_multiplier == 0.0


class TestCalculateBinaryLogprobScore:
    """Tests for calculate_binary_logprob_score function.

    This function calculates relevance score from Yes/No log probabilities.
    Implements MAIN-RAG formula: score = log P(Yes) - log P(No)
    """

    def _create_mock_response_with_logprobs(
        self,
        content: str,
        logprobs_content: list[dict],
    ) -> MagicMock:
        """Helper to create mock LLM response with logprobs metadata."""
        response = MagicMock()
        response.content = content
        response.response_metadata = {
            "logprobs": {
                "content": logprobs_content,
            }
        }
        return response

    def test_score_with_logprobs_available(self):
        """Test score calculation when both Yes and No logprobs are available."""
        from autorag_research.pipelines.generation.main_rag import calculate_binary_logprob_score

        logprobs_content = [
            {
                "token": "Yes",
                "logprob": -0.001,
                "bytes": [89, 101, 115],
                "top_logprobs": [
                    {"token": "Yes", "logprob": -0.001, "bytes": [89, 101, 115]},
                    {"token": "No", "logprob": -6.5, "bytes": [78, 111]},
                ],
            },
        ]
        response = self._create_mock_response_with_logprobs("Yes", logprobs_content)

        score, used_logprobs = calculate_binary_logprob_score(response)

        # score = log P(Yes) - log P(No) = -0.001 - (-6.5) = 6.499
        assert used_logprobs is True
        assert abs(score - 6.499) < 0.001

    def test_score_negative_when_no_is_more_likely(self):
        """Test that score is negative when No is more likely than Yes."""
        from autorag_research.pipelines.generation.main_rag import calculate_binary_logprob_score

        logprobs_content = [
            {
                "token": "No",
                "logprob": -0.01,
                "bytes": [78, 111],
                "top_logprobs": [
                    {"token": "No", "logprob": -0.01, "bytes": [78, 111]},
                    {"token": "Yes", "logprob": -4.0, "bytes": [89, 101, 115]},
                ],
            },
        ]
        response = self._create_mock_response_with_logprobs("No", logprobs_content)

        score, used_logprobs = calculate_binary_logprob_score(response)

        # score = log P(Yes) - log P(No) = -4.0 - (-0.01) = -3.99
        assert used_logprobs is True
        assert score < 0
        assert abs(score - (-3.99)) < 0.001

    def test_fallback_yes_response(self):
        """Test fallback scoring when response starts with Yes but no logprobs."""
        from autorag_research.pipelines.generation.main_rag import calculate_binary_logprob_score

        response = MagicMock()
        response.content = "Yes, the document is relevant."
        response.response_metadata = {}

        score, used_logprobs = calculate_binary_logprob_score(response)

        assert used_logprobs is False
        assert score == 1.0

    def test_fallback_no_response(self):
        """Test fallback scoring when response starts with No but no logprobs."""
        from autorag_research.pipelines.generation.main_rag import calculate_binary_logprob_score

        response = MagicMock()
        response.content = "No, the document is not relevant."
        response.response_metadata = {}

        score, used_logprobs = calculate_binary_logprob_score(response)

        assert used_logprobs is False
        assert score == -1.0

    def test_fallback_ambiguous_response(self):
        """Test fallback scoring when response is ambiguous (neither Yes nor No)."""
        from autorag_research.pipelines.generation.main_rag import calculate_binary_logprob_score

        response = MagicMock()
        response.content = "The document might be relevant in some contexts."
        response.response_metadata = {}

        score, used_logprobs = calculate_binary_logprob_score(response)

        assert used_logprobs is False
        assert score == 0.0

    def test_fallback_with_string_response(self):
        """Test fallback scoring when response is a plain string."""
        from autorag_research.pipelines.generation.main_rag import calculate_binary_logprob_score

        score, used_logprobs = calculate_binary_logprob_score("Yes")

        assert used_logprobs is False
        assert score == 1.0

    def test_fallback_case_insensitive(self):
        """Test that fallback matching is case-insensitive."""
        from autorag_research.pipelines.generation.main_rag import calculate_binary_logprob_score

        response = MagicMock()
        response.content = "YES"
        response.response_metadata = {}

        score, used_logprobs = calculate_binary_logprob_score(response)

        assert used_logprobs is False
        assert score == 1.0

    def test_custom_tokens(self):
        """Test scoring with custom positive/negative tokens."""
        from autorag_research.pipelines.generation.main_rag import calculate_binary_logprob_score

        logprobs_content = [
            {
                "token": "True",
                "logprob": -0.05,
                "bytes": [84, 114, 117, 101],
                "top_logprobs": [
                    {"token": "True", "logprob": -0.05, "bytes": [84, 114, 117, 101]},
                    {"token": "False", "logprob": -3.0, "bytes": [70, 97, 108, 115, 101]},
                ],
            },
        ]
        response = self._create_mock_response_with_logprobs("True", logprobs_content)

        score, used_logprobs = calculate_binary_logprob_score(
            response,
            positive_token="True",  # noqa: S106
            negative_token="False",  # noqa: S106
        )

        # score = log P(True) - log P(False) = -0.05 - (-3.0) = 2.95
        assert used_logprobs is True
        assert abs(score - 2.95) < 0.001

    def test_fallback_when_only_one_token_in_logprobs(self):
        """Test fallback when logprobs exist but only one of Yes/No is present."""
        from autorag_research.pipelines.generation.main_rag import calculate_binary_logprob_score

        logprobs_content = [
            {
                "token": "Yes",
                "logprob": -0.001,
                "bytes": [89, 101, 115],
                "top_logprobs": [
                    {"token": "Yes", "logprob": -0.001, "bytes": [89, 101, 115]},
                    # No "No" token in top_logprobs
                    {"token": "Sure", "logprob": -5.0, "bytes": [83, 117, 114, 101]},
                ],
            },
        ]
        response = self._create_mock_response_with_logprobs("Yes", logprobs_content)

        score, used_logprobs = calculate_binary_logprob_score(response)

        # Should fall back to text-based scoring since No is not found
        assert used_logprobs is False
        assert score == 1.0

    def test_zero_score_when_both_tokens_equal_probability(self):
        """Test that score is zero when Yes and No have equal probability."""
        from autorag_research.pipelines.generation.main_rag import calculate_binary_logprob_score

        logprobs_content = [
            {
                "token": "Yes",
                "logprob": -1.0,
                "bytes": [89, 101, 115],
                "top_logprobs": [
                    {"token": "Yes", "logprob": -1.0, "bytes": [89, 101, 115]},
                    {"token": "No", "logprob": -1.0, "bytes": [78, 111]},
                ],
            },
        ]
        response = self._create_mock_response_with_logprobs("Yes", logprobs_content)

        score, used_logprobs = calculate_binary_logprob_score(response)

        assert used_logprobs is True
        assert score == 0.0


@pytest.mark.api
class TestLogprobsWithOpenAI:
    """Integration tests for logprobs extraction with real OpenAI API.

    These tests verify that:
    1. OpenAI models return logprobs when configured with .bind(logprobs=True)
    2. extract_token_logprobs correctly extracts logprobs from OpenAI responses
    3. calculate_binary_logprob_score works with real OpenAI logprobs

    Requires: OPENAI_API_KEY environment variable
    """

    @pytest.fixture
    def openai_llm_with_logprobs(self):
        """Create ChatOpenAI instance with logprobs enabled."""
        pytest.importorskip("langchain_openai", reason="langchain-openai not installed")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
        ).bind(logprobs=True, top_logprobs=5)

    def test_openai_returns_logprobs(self, openai_llm_with_logprobs):
        """Test that OpenAI model actually returns logprobs in response_metadata."""
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content="Answer with only 'Yes' or 'No'."),
            HumanMessage(content="Is the sky blue?"),
        ]

        response = openai_llm_with_logprobs.invoke(messages)

        # Verify response_metadata contains logprobs
        assert "logprobs" in response.response_metadata
        assert response.response_metadata["logprobs"] is not None
        assert "content" in response.response_metadata["logprobs"]
        assert len(response.response_metadata["logprobs"]["content"]) > 0

        # Verify logprobs structure
        first_token = response.response_metadata["logprobs"]["content"][0]
        assert "token" in first_token
        assert "logprob" in first_token
        assert "top_logprobs" in first_token

    def test_extract_token_logprobs_with_openai(self, openai_llm_with_logprobs):
        """Test extract_token_logprobs correctly extracts Yes/No logprobs from OpenAI."""
        from langchain_core.messages import HumanMessage, SystemMessage

        from autorag_research.util import extract_token_logprobs

        messages = [
            SystemMessage(content="Answer with only 'Yes' or 'No'."),
            HumanMessage(content="Is the sky blue?"),
        ]

        response = openai_llm_with_logprobs.invoke(messages)

        # Extract Yes/No logprobs
        logprobs = extract_token_logprobs(response, target_tokens=["Yes", "No"])

        # Should find at least one of Yes or No
        assert logprobs is not None
        assert len(logprobs) > 0

        # At least one target token should be found
        found_tokens = [t for t in logprobs if t.lower() in ["yes", "no"]]
        assert len(found_tokens) > 0

        # Logprob values should be negative (log probabilities)
        for _, logprob in logprobs.items():
            assert isinstance(logprob, float)
            assert logprob <= 0  # Log probabilities are always <= 0

    def test_calculate_binary_logprob_score_with_openai(self, openai_llm_with_logprobs):
        """Test calculate_binary_logprob_score with real OpenAI logprobs."""
        from langchain_core.messages import HumanMessage, SystemMessage

        from autorag_research.pipelines.generation.main_rag import calculate_binary_logprob_score

        messages = [
            SystemMessage(content="Answer with only 'Yes' or 'No'."),
            HumanMessage(content="Is the sky blue?"),
        ]

        response = openai_llm_with_logprobs.invoke(messages)

        score, used_logprobs = calculate_binary_logprob_score(response)

        # Should use actual logprobs, not fallback
        assert used_logprobs is True, "Should use real logprobs from OpenAI"

        # Score should be a float
        assert isinstance(score, float)

        # Score is log P(Yes) - log P(No), can be positive or negative
        # Just verify it's a reasonable value (not NaN, not infinite)
        assert -100 < score < 100, f"Score should be reasonable, got {score}"

    def test_logprob_scores_differ_for_different_questions(self, openai_llm_with_logprobs):
        """Test that logprob scores differ for different questions."""
        from langchain_core.messages import HumanMessage, SystemMessage

        from autorag_research.pipelines.generation.main_rag import calculate_binary_logprob_score

        # Question 1: Should favor Yes
        messages1 = [
            SystemMessage(content="Answer with only 'Yes' or 'No'."),
            HumanMessage(content="Is the sky blue during a clear day?"),
        ]
        response1 = openai_llm_with_logprobs.invoke(messages1)
        score1, used1 = calculate_binary_logprob_score(response1)

        # Question 2: Should favor No
        messages2 = [
            SystemMessage(content="Answer with only 'Yes' or 'No'."),
            HumanMessage(content="Is the sky green?"),
        ]
        response2 = openai_llm_with_logprobs.invoke(messages2)
        score2, used2 = calculate_binary_logprob_score(response2)

        # Both should use real logprobs
        assert used1 is True, "Question 1 should use real logprobs"
        assert used2 is True, "Question 2 should use real logprobs"

        # Scores should be different (model gives different confidence for different questions)
        assert score1 != score2, f"Scores should differ: {score1} vs {score2}"

        # Score for "blue sky" should be higher (more Yes-biased) than "green sky"
        assert score1 > score2, f"Blue sky score ({score1}) should be higher than green sky ({score2})"

    def test_openai_without_logprobs_falls_back(self):
        """Test that OpenAI without logprobs enabled falls back to text-based scoring."""
        pytest.importorskip("langchain_openai", reason="langchain-openai not installed")
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI

        from autorag_research.pipelines.generation.main_rag import calculate_binary_logprob_score

        # Create LLM WITHOUT logprobs
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        messages = [
            SystemMessage(content="Answer with only 'Yes' or 'No'."),
            HumanMessage(content="Is the sky blue?"),
        ]

        response = llm.invoke(messages)

        score, used_logprobs = calculate_binary_logprob_score(response)

        # Should fall back to text-based scoring
        assert used_logprobs is False, "Should use fallback when logprobs not enabled"
        assert score in [1.0, -1.0, 0.0], "Fallback score should be 1.0, -1.0, or 0.0"
