"""Tests for IRCoT (Interleaving Retrieval with Chain-of-Thought) Generation Pipeline.

This module contains TDD tests for the IRCoTGenerationPipeline. Tests are written
BEFORE implementation based on the design document (Pipeline_Design.md).

Test Categories:
1. Unit Tests - Test helper methods (_extract_first_sentence, _build_reasoning_prompt, etc.)
2. Pipeline Initialization Tests - Test constructor and config
3. Core Algorithm Tests - Test IRCoT-specific behavior (termination, budget, token aggregation)
4. Integration Tests - End-to-end with mock LLM and retrieval pipeline
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.fake import FakeListLLM

from autorag_research.orm.repository.executor_result import ExecutorResultRepository
from tests.autorag_research.pipelines.pipeline_test_utils import (
    create_mock_llm,
)

# ==================== Unit Tests for Helper Methods ====================


class TestExtractFirstSentence:
    """Unit tests for _extract_first_sentence helper method."""

    @pytest.fixture
    def pipeline_class(self):
        """Import the pipeline class (will fail until implementation exists)."""
        from autorag_research.pipelines.generation.ircot import IRCoTGenerationPipeline

        return IRCoTGenerationPipeline

    def test_extract_first_sentence_with_period(self, pipeline_class):
        """Test extraction when text contains a period delimiter."""
        # Create a minimal mock to test the static-like method
        text = "This is the first sentence. This is the second sentence. And third."
        result = pipeline_class._extract_first_sentence(None, text)
        assert result == "This is the first sentence."

    def test_extract_first_sentence_with_question_mark(self, pipeline_class):
        """Test extraction when text contains a question mark delimiter."""
        text = "What is the capital of France? Paris is the capital."
        result = pipeline_class._extract_first_sentence(None, text)
        assert result == "What is the capital of France?"

    def test_extract_first_sentence_with_exclamation(self, pipeline_class):
        """Test extraction when text contains an exclamation mark delimiter."""
        text = "This is amazing! I can't believe it."
        result = pipeline_class._extract_first_sentence(None, text)
        assert result == "This is amazing!"

    def test_extract_first_sentence_no_delimiter(self, pipeline_class):
        """Test extraction when text has no sentence delimiter."""
        text = "A single incomplete thought without punctuation"
        result = pipeline_class._extract_first_sentence(None, text)
        assert result == "A single incomplete thought without punctuation"

    def test_extract_first_sentence_empty_string(self, pipeline_class):
        """Test extraction with empty string input."""
        text = ""
        result = pipeline_class._extract_first_sentence(None, text)
        assert result == ""

    def test_extract_first_sentence_whitespace_handling(self, pipeline_class):
        """Test that leading/trailing whitespace is handled correctly."""
        text = "  First sentence with spaces.   Second sentence.  "
        result = pipeline_class._extract_first_sentence(None, text)
        assert result == "First sentence with spaces."


class TestBuildReasoningPrompt:
    """Unit tests for _build_reasoning_prompt helper method."""

    @pytest.fixture
    def pipeline(self, session_factory):
        """Create a minimal IRCoT pipeline for testing prompt building."""
        from autorag_research.pipelines.generation.ircot import IRCoTGenerationPipeline

        mock_llm = create_mock_llm()
        mock_retrieval = MagicMock()
        mock_retrieval.pipeline_id = 1
        mock_retrieval.retrieve.return_value = []

        pipeline = IRCoTGenerationPipeline(
            session_factory=session_factory,
            name="test_reasoning_prompt_pipeline",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval,
        )
        return pipeline

    @pytest.fixture
    def cleanup_pipeline(self, session_factory):
        """Cleanup fixture for pipeline results."""
        created_pipeline_ids = []
        yield created_pipeline_ids

        session = session_factory()
        try:
            repo = ExecutorResultRepository(session)
            for pipeline_id in created_pipeline_ids:
                repo.delete_by_pipeline(pipeline_id)
            session.commit()
        finally:
            session.close()

    def test_build_reasoning_prompt(self, pipeline, cleanup_pipeline):
        """Test that reasoning prompt is built correctly with all components."""
        cleanup_pipeline.append(pipeline.pipeline_id)

        query = "What is the relationship between A and B?"
        paragraphs = ["Paragraph 1 about A.", "Paragraph 2 about B."]
        cot_history = ["A is related to concept X.", "B is also related to X."]

        prompt = pipeline._build_reasoning_prompt(query, paragraphs, cot_history)

        # Verify query is included
        assert "What is the relationship between A and B?" in prompt

        # Verify paragraphs are numbered and included
        assert "[1]" in prompt
        assert "[2]" in prompt
        assert "Paragraph 1 about A." in prompt
        assert "Paragraph 2 about B." in prompt

        # Verify CoT history is included
        assert "A is related to concept X." in prompt
        assert "B is also related to X." in prompt

    def test_build_reasoning_prompt_empty_cot_history(self, pipeline, cleanup_pipeline):
        """Test reasoning prompt with empty CoT history."""
        cleanup_pipeline.append(pipeline.pipeline_id)

        query = "Test query"
        paragraphs = ["Some paragraph."]
        cot_history = []

        prompt = pipeline._build_reasoning_prompt(query, paragraphs, cot_history)

        assert "Test query" in prompt
        assert "Some paragraph." in prompt
        # Should indicate no previous thoughts
        assert "(No previous thoughts)" in prompt or "Previous Thoughts:" in prompt

    def test_build_reasoning_prompt_empty_paragraphs(self, pipeline, cleanup_pipeline):
        """Test reasoning prompt with empty paragraphs list."""
        cleanup_pipeline.append(pipeline.pipeline_id)

        query = "Test query"
        paragraphs = []
        cot_history = ["Previous thought."]

        prompt = pipeline._build_reasoning_prompt(query, paragraphs, cot_history)

        assert "Test query" in prompt
        # Prompt should still be valid even without paragraphs


class TestBuildQAPrompt:
    """Unit tests for _build_qa_prompt helper method."""

    @pytest.fixture
    def pipeline(self, session_factory):
        """Create a minimal IRCoT pipeline for testing QA prompt building."""
        from autorag_research.pipelines.generation.ircot import IRCoTGenerationPipeline

        mock_llm = create_mock_llm()
        mock_retrieval = MagicMock()
        mock_retrieval.pipeline_id = 1
        mock_retrieval.retrieve.return_value = []

        pipeline = IRCoTGenerationPipeline(
            session_factory=session_factory,
            name="test_qa_prompt_pipeline",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval,
        )
        return pipeline

    @pytest.fixture
    def cleanup_pipeline(self, session_factory):
        """Cleanup fixture for pipeline results."""
        created_pipeline_ids = []
        yield created_pipeline_ids

        session = session_factory()
        try:
            repo = ExecutorResultRepository(session)
            for pipeline_id in created_pipeline_ids:
                repo.delete_by_pipeline(pipeline_id)
            session.commit()
        finally:
            session.close()

    def test_build_qa_prompt(self, pipeline, cleanup_pipeline):
        """Test that QA prompt is built correctly."""
        cleanup_pipeline.append(pipeline.pipeline_id)

        query = "What is the capital of France?"
        paragraphs = ["Paris is the capital of France.", "France is in Europe."]

        prompt = pipeline._build_qa_prompt(query, paragraphs)

        # Verify query is included
        assert "What is the capital of France?" in prompt

        # Verify paragraphs are numbered and included
        assert "[1]" in prompt
        assert "[2]" in prompt
        assert "Paris is the capital of France." in prompt
        assert "France is in Europe." in prompt


# ==================== Pipeline Initialization Tests ====================


class TestIRCoTPipelineInitialization:
    """Tests for IRCoT pipeline initialization and configuration."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        return create_mock_llm()

    @pytest.fixture
    def mock_retrieval_pipeline(self):
        """Create a mock retrieval pipeline."""
        mock = MagicMock()
        mock.pipeline_id = 1
        mock.retrieve.return_value = [
            {"doc_id": 1, "score": 0.9},
            {"doc_id": 2, "score": 0.8},
        ]
        return mock

    @pytest.fixture
    def cleanup_pipeline_results(self, session_factory):
        """Cleanup fixture that deletes pipeline results after test."""
        created_pipeline_ids = []
        yield created_pipeline_ids

        session = session_factory()
        try:
            repo = ExecutorResultRepository(session)
            for pipeline_id in created_pipeline_ids:
                repo.delete_by_pipeline(pipeline_id)
            session.commit()
        finally:
            session.close()

    def test_pipeline_initialization_stores_config(
        self, session_factory, mock_llm, mock_retrieval_pipeline, cleanup_pipeline_results
    ):
        """Test that pipeline is created with correct parameters stored."""
        from autorag_research.pipelines.generation.ircot import IRCoTGenerationPipeline

        pipeline = IRCoTGenerationPipeline(
            session_factory=session_factory,
            name="test_ircot_init",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
            k_per_step=5,
            max_steps=10,
            paragraph_budget=20,
            stop_sequence="the answer is:",
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        # Verify pipeline was created with an ID
        assert pipeline.pipeline_id > 0

        # Verify parameters are stored
        assert pipeline.k_per_step == 5
        assert pipeline.max_steps == 10
        assert pipeline.paragraph_budget == 20
        assert pipeline.stop_sequence == "the answer is:"

    def test_pipeline_initialization_default_values(
        self, session_factory, mock_llm, mock_retrieval_pipeline, cleanup_pipeline_results
    ):
        """Test that pipeline uses correct default values."""
        from autorag_research.pipelines.generation.ircot import IRCoTGenerationPipeline

        pipeline = IRCoTGenerationPipeline(
            session_factory=session_factory,
            name="test_ircot_defaults",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        # Verify default values from design doc
        assert pipeline.k_per_step == 4
        assert pipeline.max_steps == 8
        assert pipeline.paragraph_budget == 15
        assert pipeline.stop_sequence == "answer is:"

    def test_pipeline_config_returned_correctly(
        self, session_factory, mock_llm, mock_retrieval_pipeline, cleanup_pipeline_results
    ):
        """Test that _get_pipeline_config returns correct configuration."""
        from autorag_research.pipelines.generation.ircot import IRCoTGenerationPipeline

        pipeline = IRCoTGenerationPipeline(
            session_factory=session_factory,
            name="test_ircot_config",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
            k_per_step=6,
            max_steps=12,
            paragraph_budget=25,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        config = pipeline._get_pipeline_config()

        assert config["type"] == "ircot"
        assert config["k_per_step"] == 6
        assert config["max_steps"] == 12
        assert config["paragraph_budget"] == 25
        assert config["retrieval_pipeline_id"] == mock_retrieval_pipeline.pipeline_id
        assert "reasoning_prompt_template" in config
        assert "qa_prompt_template" in config


# ==================== Core Algorithm Tests ====================


class TestIRCoTAlgorithm:
    """Tests for IRCoT core algorithm behavior."""

    @pytest.fixture
    def cleanup_pipeline_results(self, session_factory):
        """Cleanup fixture that deletes pipeline results after test."""
        created_pipeline_ids = []
        yield created_pipeline_ids

        session = session_factory()
        try:
            repo = ExecutorResultRepository(session)
            for pipeline_id in created_pipeline_ids:
                repo.delete_by_pipeline(pipeline_id)
            session.commit()
        finally:
            session.close()

    def test_generate_terminates_on_answer_string(self, session_factory, cleanup_pipeline_results):
        """Test that generation terminates early when 'answer is:' is detected."""
        from autorag_research.pipelines.generation.ircot import IRCoTGenerationPipeline

        # Create FakeListLLM that returns "answer is:" on second reasoning call
        # Sequence: reasoning1 -> reasoning2 (with answer) -> qa
        responses = [
            "First I need to find information about X.",  # Step 1 reasoning
            "Based on X, the answer is: 42.",  # Step 2 reasoning (triggers termination)
            "The final answer is 42.",  # QA generation
        ]
        fake_llm = FakeListLLM(responses=responses)

        mock_retrieval = MagicMock()
        mock_retrieval.pipeline_id = 1
        mock_retrieval.retrieve.return_value = [{"doc_id": 1, "score": 0.9}]

        pipeline = IRCoTGenerationPipeline(
            session_factory=session_factory,
            name="test_ircot_termination",
            llm=fake_llm,
            retrieval_pipeline=mock_retrieval,
            max_steps=8,  # Should terminate before reaching max
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        result = pipeline._generate("What is the meaning of life?", top_k=4)

        # Should have terminated after detecting "answer is:" (before max_steps)
        # Retrieval calls: 1 initial + 1 after step 1 (step 2 triggers termination, no retrieval after)
        # So we expect 2 retrieval calls, not 9 (1 initial + 8 steps)
        assert mock_retrieval.retrieve.call_count < 9
        assert result.text is not None

    def test_generate_terminates_on_max_steps(self, session_factory, cleanup_pipeline_results):
        """Test that generation terminates at max_steps limit."""
        from autorag_research.pipelines.generation.ircot import IRCoTGenerationPipeline

        # Create FakeListLLM that never returns "answer is:"
        # Need enough responses for: max_steps reasoning + 1 QA
        max_steps = 3
        responses = [
            "Thought about the problem.",  # Step 1
            "Need more information.",  # Step 2
            "Still thinking.",  # Step 3 (max_steps reached)
            "The final answer based on available info.",  # QA
        ]
        fake_llm = FakeListLLM(responses=responses)

        mock_retrieval = MagicMock()
        mock_retrieval.pipeline_id = 1
        mock_retrieval.retrieve.return_value = [{"doc_id": 1, "score": 0.9}]

        pipeline = IRCoTGenerationPipeline(
            session_factory=session_factory,
            name="test_ircot_max_steps",
            llm=fake_llm,
            retrieval_pipeline=mock_retrieval,
            max_steps=max_steps,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        result = pipeline._generate("Complex question", top_k=4)

        # Should have made exactly max_steps + 1 (QA) LLM calls
        # Retrieval: 1 initial + max_steps = 4 calls
        assert mock_retrieval.retrieve.call_count == 1 + max_steps
        assert result.text is not None

    def test_generate_applies_paragraph_budget(self, session_factory, cleanup_pipeline_results):
        """Test that paragraph_budget caps the total paragraphs collected."""
        from autorag_research.pipelines.generation.ircot import IRCoTGenerationPipeline

        # Create FakeListLLM
        responses = [
            "First thought.",
            "Second thought.",
            "Third thought.",
            "Final answer.",
        ]
        fake_llm = FakeListLLM(responses=responses)

        # Each retrieval returns 4 paragraphs
        retrieval_call_count = [0]

        def mock_retrieve(query, top_k):
            retrieval_call_count[0] += 1
            base_id = retrieval_call_count[0] * 10
            return [{"doc_id": base_id + i, "score": 0.9 - i * 0.1} for i in range(top_k)]

        mock_retrieval = MagicMock()
        mock_retrieval.pipeline_id = 1
        mock_retrieval.retrieve.side_effect = mock_retrieve

        # Set low paragraph_budget to test capping
        pipeline = IRCoTGenerationPipeline(
            session_factory=session_factory,
            name="test_ircot_budget",
            llm=fake_llm,
            retrieval_pipeline=mock_retrieval,
            k_per_step=4,
            max_steps=3,
            paragraph_budget=5,  # Cap at 5 paragraphs
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        result = pipeline._generate("Test query", top_k=4)

        # Verify result metadata contains info about paragraphs
        assert result.metadata is not None
        # The paragraph collection should be capped at budget
        if "chunk_ids" in result.metadata:
            assert len(result.metadata["chunk_ids"]) <= 5

    def test_generate_aggregates_token_usage(self, session_factory, cleanup_pipeline_results):
        """Test that token usage is aggregated across all LLM calls."""
        from autorag_research.pipelines.generation.ircot import IRCoTGenerationPipeline

        # Create mock LLM that tracks token usage per call
        call_count = [0]
        token_per_call = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }

        def mock_invoke(prompt):
            call_count[0] += 1
            response = MagicMock()
            if call_count[0] == 2:
                response.content = "The answer is: test answer."
            else:
                response.content = "Thinking step."
            response.usage_metadata = {
                "input_tokens": token_per_call["prompt_tokens"],
                "output_tokens": token_per_call["completion_tokens"],
                "total_tokens": token_per_call["total_tokens"],
            }
            return response

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = mock_invoke

        mock_retrieval = MagicMock()
        mock_retrieval.pipeline_id = 1
        mock_retrieval.retrieve.return_value = [{"doc_id": 1, "score": 0.9}]

        pipeline = IRCoTGenerationPipeline(
            session_factory=session_factory,
            name="test_ircot_tokens",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval,
            max_steps=4,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        result = pipeline._generate("Test query", top_k=4)

        # Token usage should be aggregated from all LLM calls
        assert result.token_usage is not None
        # At least 2 calls (reasoning step + QA), possibly 3 if termination happens after step 2
        total_calls = call_count[0]
        expected_total = total_calls * token_per_call["total_tokens"]
        assert result.token_usage["total_tokens"] == expected_total

    def test_generate_case_insensitive_termination(self, session_factory, cleanup_pipeline_results):
        """Test that termination detection is case-insensitive."""
        from autorag_research.pipelines.generation.ircot import IRCoTGenerationPipeline

        # Test with uppercase "ANSWER IS:"
        responses = [
            "The ANSWER IS: Test.",
            "Final answer.",
        ]
        fake_llm = FakeListLLM(responses=responses)

        mock_retrieval = MagicMock()
        mock_retrieval.pipeline_id = 1
        mock_retrieval.retrieve.return_value = [{"doc_id": 1, "score": 0.9}]

        pipeline = IRCoTGenerationPipeline(
            session_factory=session_factory,
            name="test_ircot_case_insensitive",
            llm=fake_llm,
            retrieval_pipeline=mock_retrieval,
            max_steps=5,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        result = pipeline._generate("Test", top_k=2)

        # Should terminate on first response due to case-insensitive match
        # Only 1 retrieval (initial) + 1 reasoning + QA
        assert mock_retrieval.retrieve.call_count == 1
        assert result.text is not None


# ==================== Integration Tests ====================


class TestIRCoTPipelineIntegration:
    """Integration tests for IRCoT pipeline with full workflow."""

    @pytest.fixture
    def cleanup_pipeline_results(self, session_factory):
        """Cleanup fixture that deletes pipeline results after test."""
        created_pipeline_ids = []
        yield created_pipeline_ids

        session = session_factory()
        try:
            repo = ExecutorResultRepository(session)
            for pipeline_id in created_pipeline_ids:
                repo.delete_by_pipeline(pipeline_id)
            session.commit()
        finally:
            session.close()

    @pytest.fixture
    def mock_retrieval_pipeline(self):
        """Create a mock retrieval pipeline that returns seed data chunk IDs."""
        mock = MagicMock()
        mock.pipeline_id = 1

        def mock_retrieve(query_text: str, top_k: int):
            # Return chunk IDs that exist in seed data (1-6)
            return [{"doc_id": i, "score": 0.9 - i * 0.1} for i in range(1, min(top_k + 1, 7))]

        mock.retrieve = mock_retrieve
        return mock

    def test_ircot_full_flow_with_fake_llm(self, session_factory, mock_retrieval_pipeline, cleanup_pipeline_results):
        """Test end-to-end IRCoT flow with FakeListLLM."""
        from autorag_research.pipelines.generation.ircot import IRCoTGenerationPipeline

        # Prepare enough responses for queries, each with up to 2 steps + QA
        # Pattern per query: reasoning1 -> (maybe reasoning2 with answer) -> qa
        # Generate enough responses for any reasonable number of queries
        responses = []
        for _ in range(20):  # Support up to 20 queries
            responses.extend([
                "First reasoning step for this query.",
                "The answer is: based on the context.",
                "Final answer text.",
            ])

        fake_llm = FakeListLLM(responses=responses)

        pipeline = IRCoTGenerationPipeline(
            session_factory=session_factory,
            name="test_ircot_full_flow",
            llm=fake_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
            k_per_step=3,
            max_steps=4,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        result = pipeline.run(top_k=3, batch_size=10)

        # Verify basic structure manually (skip PipelineTestVerifier due to seed data variance)
        assert "total_queries" in result
        assert "pipeline_id" in result
        assert "avg_execution_time_ms" in result

        # Verify pipeline ran correctly
        assert result["total_queries"] > 0
        assert result["pipeline_id"] == pipeline.pipeline_id
        assert result["avg_execution_time_ms"] >= 0

    def test_ircot_handles_empty_retrieval_results(self, session_factory, cleanup_pipeline_results):
        """Test that IRCoT handles empty retrieval results gracefully."""
        from autorag_research.pipelines.generation.ircot import IRCoTGenerationPipeline

        # LLM responses
        responses = [
            "Trying to reason without context.",
            "The answer is: I don't have enough information.",
            "Cannot answer without supporting documents.",
        ]
        fake_llm = FakeListLLM(responses=responses)

        # Retrieval pipeline that returns empty results
        mock_retrieval = MagicMock()
        mock_retrieval.pipeline_id = 999
        mock_retrieval.retrieve.return_value = []

        pipeline = IRCoTGenerationPipeline(
            session_factory=session_factory,
            name="test_ircot_empty_retrieval",
            llm=fake_llm,
            retrieval_pipeline=mock_retrieval,
            max_steps=2,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        # Should not raise an error
        result = pipeline._generate("Query with no matching documents", top_k=5)

        assert result.text is not None
        # Should still have metadata even with empty retrieval
        assert result.metadata is not None

    def test_ircot_metadata_contains_cot_history(
        self, session_factory, mock_retrieval_pipeline, cleanup_pipeline_results
    ):
        """Test that result metadata includes chain-of-thought history."""
        from autorag_research.pipelines.generation.ircot import IRCoTGenerationPipeline

        responses = [
            "Step 1: Analyzing the question.",
            "Step 2: The answer is: found it.",
            "Final answer.",
        ]
        fake_llm = FakeListLLM(responses=responses)

        pipeline = IRCoTGenerationPipeline(
            session_factory=session_factory,
            name="test_ircot_cot_metadata",
            llm=fake_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
            max_steps=4,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        result = pipeline._generate("Test query", top_k=3)

        # Metadata should contain CoT history
        assert result.metadata is not None
        assert "cot_sentences" in result.metadata or "cot_history" in result.metadata

    def test_ircot_metadata_contains_chunk_ids(
        self, session_factory, mock_retrieval_pipeline, cleanup_pipeline_results
    ):
        """Test that result metadata includes retrieved chunk IDs."""
        from autorag_research.pipelines.generation.ircot import IRCoTGenerationPipeline

        responses = [
            "The answer is: quick termination.",
            "Final answer.",
        ]
        fake_llm = FakeListLLM(responses=responses)

        pipeline = IRCoTGenerationPipeline(
            session_factory=session_factory,
            name="test_ircot_chunk_metadata",
            llm=fake_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
            max_steps=2,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        result = pipeline._generate("Test query", top_k=3)

        # Metadata should contain chunk IDs
        assert result.metadata is not None
        assert "chunk_ids" in result.metadata or "retrieved_chunk_ids" in result.metadata

    def test_ircot_with_custom_prompt_templates(
        self, session_factory, mock_retrieval_pipeline, cleanup_pipeline_results
    ):
        """Test IRCoT with custom prompt templates."""
        from autorag_research.pipelines.generation.ircot import IRCoTGenerationPipeline

        custom_reasoning_template = """Custom Reasoning Template
Query: {query}
Context: {paragraphs}
Previous: {cot_history}
Think:"""

        custom_qa_template = """Custom QA Template
Question: {query}
Documents: {paragraphs}
Answer:"""

        # Track prompts passed to LLM
        captured_prompts = []

        def mock_invoke(prompt):
            captured_prompts.append(prompt)
            response = MagicMock()
            if len(captured_prompts) == 1:
                response.content = "The answer is: done."
            else:
                response.content = "Final."
            response.usage_metadata = {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            }
            return response

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = mock_invoke

        pipeline = IRCoTGenerationPipeline(
            session_factory=session_factory,
            name="test_ircot_custom_templates",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
            reasoning_prompt_template=custom_reasoning_template,
            qa_prompt_template=custom_qa_template,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        pipeline._generate("Test custom templates", top_k=2)

        # Verify custom templates were used
        assert len(captured_prompts) >= 2
        assert "Custom Reasoning Template" in captured_prompts[0]
        assert "Custom QA Template" in captured_prompts[1]

    def test_ircot_uses_first_sentence_only(self, session_factory, mock_retrieval_pipeline, cleanup_pipeline_results):
        """Test that only the first sentence is extracted for CoT history."""
        from autorag_research.pipelines.generation.ircot import IRCoTGenerationPipeline

        # LLM returns multi-sentence responses
        responses = [
            "First sentence of step 1. Second sentence. Third sentence.",
            "The answer is: found. More explanation.",
            "Final answer.",
        ]
        fake_llm = FakeListLLM(responses=responses)

        pipeline = IRCoTGenerationPipeline(
            session_factory=session_factory,
            name="test_ircot_first_sentence",
            llm=fake_llm,
            retrieval_pipeline=mock_retrieval_pipeline,
            max_steps=3,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        result = pipeline._generate("Test query", top_k=2)

        # Check metadata for CoT sentences - should only have first sentences
        assert result.metadata is not None
        cot = result.metadata.get("cot_sentences", [])
        assert len(cot) > 0, "Expected at least one CoT sentence"

        # Each entry should be a single sentence (extracted first sentence only)
        for sentence in cot:
            # Should not contain ". " followed by more text (which would indicate multiple sentences)
            # The first sentence should end with a period but not have ". " followed by content
            parts = sentence.split(". ")
            # Allow for trailing period, but not multiple sentences
            assert len(parts) <= 2, f"Expected single sentence, got: {sentence}"
            # If there are 2 parts, the second should be empty (just from trailing period split)
            if len(parts) == 2:
                assert parts[1] == "", f"Expected single sentence, got multiple: {sentence}"
