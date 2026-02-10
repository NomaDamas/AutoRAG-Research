"""Tests for IRCoT (Interleaving Retrieval with Chain-of-Thought) Generation Pipeline.

Test Categories:
1. Unit Tests - Helper methods (_extract_first_sentence, _build_reasoning_prompt, etc.)
2. Pipeline Initialization Tests - Constructor and config
3. Core Algorithm Tests - IRCoT-specific behavior (termination, budget, token aggregation)
4. Integration Tests - End-to-end with PipelineTestVerifier
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.language_models.fake import FakeListLLM

from tests.autorag_research.pipelines.pipeline_test_utils import (
    PipelineTestConfig,
    PipelineTestVerifier,
    cleanup_pipeline_results_factory,
    create_mock_llm,
    create_mock_retrieval_pipeline,
)

# ==================== Shared Fixtures ====================


@pytest.fixture
def cleanup(session_factory):
    """Cleanup fixture for pipeline results."""
    yield from cleanup_pipeline_results_factory(session_factory)


@pytest.fixture
def mock_retrieval():
    """Create a mock retrieval pipeline."""
    return create_mock_retrieval_pipeline()


@pytest.fixture
def ircot_pipeline(session_factory, cleanup):
    """Create an IRCoT pipeline with default settings for testing."""
    from autorag_research.pipelines.generation.ircot import IRCoTGenerationPipeline

    pipeline = IRCoTGenerationPipeline(
        session_factory=session_factory,
        name="test_ircot_fixture",
        llm=create_mock_llm(),
        retrieval_pipeline=create_mock_retrieval_pipeline(),
    )
    cleanup.append(pipeline.pipeline_id)
    return pipeline


# ==================== Unit Tests for Helper Methods ====================


class TestExtractFirstSentence:
    """Unit tests for _extract_first_sentence helper method."""

    @pytest.fixture
    def pipeline_class(self):
        from autorag_research.pipelines.generation.ircot import IRCoTGenerationPipeline

        return IRCoTGenerationPipeline

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("First sentence. Second sentence.", "First sentence."),
            ("What is this? Answer here.", "What is this?"),
            ("Amazing! More text.", "Amazing!"),
            ("No delimiter here", "No delimiter here"),
            ("", ""),
            ("  Spaces around.  More.", "Spaces around."),
        ],
        ids=["period", "question", "exclamation", "no_delimiter", "empty", "whitespace"],
    )
    def test_extract_first_sentence(self, pipeline_class, text, expected):
        """Test first sentence extraction with various inputs."""
        result = pipeline_class._extract_first_sentence(None, text)
        assert result == expected


class TestBuildPrompts:
    """Unit tests for prompt building methods."""

    def test_build_reasoning_prompt(self, ircot_pipeline):
        """Test reasoning prompt contains all components."""
        prompt = ircot_pipeline._build_reasoning_prompt(
            query="What is A?",
            paragraphs=["Para 1.", "Para 2."],
            cot_history=["Thought 1.", "Thought 2."],
        )
        assert "What is A?" in prompt
        assert "[1]" in prompt and "[2]" in prompt
        assert "Para 1." in prompt and "Para 2." in prompt
        assert "Thought 1." in prompt and "Thought 2." in prompt

    def test_build_reasoning_prompt_empty_history(self, ircot_pipeline):
        """Test reasoning prompt with empty CoT history."""
        prompt = ircot_pipeline._build_reasoning_prompt("Query", ["Para"], [])
        assert "Query" in prompt
        assert "(No previous thoughts)" in prompt

    def test_build_qa_prompt(self, ircot_pipeline):
        """Test QA prompt is built correctly."""
        prompt = ircot_pipeline._build_qa_prompt("Question?", ["Doc 1", "Doc 2"])
        assert "Question?" in prompt
        assert "[1]" in prompt and "[2]" in prompt
        assert "Doc 1" in prompt and "Doc 2" in prompt


# ==================== Pipeline Initialization Tests ====================


class TestIRCoTPipelineInitialization:
    """Tests for IRCoT pipeline initialization and configuration."""

    def test_initialization_stores_config(self, session_factory, mock_retrieval, cleanup):
        """Test pipeline stores custom parameters correctly."""
        from autorag_research.pipelines.generation.ircot import IRCoTGenerationPipeline

        pipeline = IRCoTGenerationPipeline(
            session_factory=session_factory,
            name="test_init",
            llm=create_mock_llm(),
            retrieval_pipeline=mock_retrieval,
            k_per_step=5,
            max_steps=10,
            paragraph_budget=20,
            stop_sequence="the answer is:",
        )
        cleanup.append(pipeline.pipeline_id)

        assert pipeline.pipeline_id > 0
        assert pipeline.k_per_step == 5
        assert pipeline.max_steps == 10
        assert pipeline.paragraph_budget == 20
        assert pipeline.stop_sequence == "the answer is:"

    def test_initialization_default_values(self, session_factory, mock_retrieval, cleanup):
        """Test pipeline uses correct default values."""
        from autorag_research.pipelines.generation.ircot import IRCoTGenerationPipeline

        pipeline = IRCoTGenerationPipeline(
            session_factory=session_factory,
            name="test_defaults",
            llm=create_mock_llm(),
            retrieval_pipeline=mock_retrieval,
        )
        cleanup.append(pipeline.pipeline_id)

        assert pipeline.k_per_step == 4
        assert pipeline.max_steps == 8
        assert pipeline.paragraph_budget == 15
        assert pipeline.stop_sequence == "answer is:"

    def test_get_pipeline_config(self, session_factory, mock_retrieval, cleanup):
        """Test _get_pipeline_config returns correct configuration."""
        from autorag_research.pipelines.generation.ircot import IRCoTGenerationPipeline

        pipeline = IRCoTGenerationPipeline(
            session_factory=session_factory,
            name="test_config",
            llm=create_mock_llm(),
            retrieval_pipeline=mock_retrieval,
            k_per_step=6,
            max_steps=12,
        )
        cleanup.append(pipeline.pipeline_id)

        config = pipeline._get_pipeline_config()
        assert config["type"] == "ircot"
        assert config["k_per_step"] == 6
        assert config["max_steps"] == 12
        assert config["retrieval_pipeline_id"] == mock_retrieval.pipeline_id
        assert "reasoning_prompt_template" in config
        assert "qa_prompt_template" in config


# ==================== Core Algorithm Tests ====================


class TestIRCoTAlgorithm:
    """Tests for IRCoT core algorithm behavior."""

    @pytest.mark.asyncio
    async def test_terminates_on_answer_string(self, session_factory, cleanup):
        """Test generation terminates when 'answer is:' is detected."""
        from autorag_research.pipelines.generation.ircot import IRCoTGenerationPipeline

        # Use create_mock_llm which sets up ainvoke as AsyncMock
        mock_llm = create_mock_llm()
        call_count = [0]
        responses = [
            "First reasoning.",
            "The answer is: 42.",
            "Final answer.",
        ]

        async def mock_ainvoke(prompt):
            response = MagicMock()
            response.content = responses[min(call_count[0], len(responses) - 1)]
            response.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
            call_count[0] += 1
            return response

        mock_llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
        mock_retrieval = create_mock_retrieval_pipeline()

        pipeline = IRCoTGenerationPipeline(
            session_factory=session_factory,
            name="test_termination",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval,
            max_steps=8,
        )
        cleanup.append(pipeline.pipeline_id)

        # Use query_id=1 from seed data instead of string
        result = await pipeline._generate(1, top_k=4)

        # Verify early termination via metadata
        assert result.metadata is not None
        assert result.metadata.get("steps", 0) < 8  # Should terminate early
        assert result.text is not None

    @pytest.mark.asyncio
    async def test_terminates_on_max_steps(self, session_factory, cleanup):
        """Test generation terminates at max_steps limit."""
        from autorag_research.pipelines.generation.ircot import IRCoTGenerationPipeline

        max_steps = 3
        mock_llm = create_mock_llm()
        call_count = [0]
        responses = ["Thought 1.", "Thought 2.", "Thought 3.", "Final answer."]

        async def mock_ainvoke(prompt):
            response = MagicMock()
            response.content = responses[min(call_count[0], len(responses) - 1)]
            response.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
            call_count[0] += 1
            return response

        mock_llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
        mock_retrieval = create_mock_retrieval_pipeline()

        pipeline = IRCoTGenerationPipeline(
            session_factory=session_factory,
            name="test_max_steps",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval,
            max_steps=max_steps,
        )
        cleanup.append(pipeline.pipeline_id)

        # Use query_id=1 from seed data instead of string
        result = await pipeline._generate(1, top_k=4)

        # Verify steps completed matches max_steps
        assert result.metadata is not None
        assert result.metadata.get("steps") == max_steps
        assert result.text is not None

    @pytest.mark.asyncio
    async def test_applies_paragraph_budget(self, session_factory, cleanup):
        """Test paragraph_budget caps total paragraphs collected."""

        from autorag_research.pipelines.generation.ircot import IRCoTGenerationPipeline

        mock_llm = create_mock_llm()
        llm_call_count = [0]
        responses = ["Thought.", "Thought.", "Thought.", "Answer."]

        async def mock_ainvoke(prompt):
            response = MagicMock()
            response.content = responses[min(llm_call_count[0], len(responses) - 1)]
            response.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
            llm_call_count[0] += 1
            return response

        mock_llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)

        retrieve_call_count = [0]

        async def mock_retrieve(query, top_k):
            retrieve_call_count[0] += 1
            return [{"doc_id": retrieve_call_count[0] * 10 + i, "score": 0.9} for i in range(top_k)]

        mock_retrieval = MagicMock()
        mock_retrieval.pipeline_id = 1
        mock_retrieval.retrieve = AsyncMock(side_effect=mock_retrieve)
        mock_retrieval._retrieve_by_id = AsyncMock(side_effect=mock_retrieve)

        pipeline = IRCoTGenerationPipeline(
            session_factory=session_factory,
            name="test_budget",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval,
            k_per_step=4,
            max_steps=3,
            paragraph_budget=5,
        )
        cleanup.append(pipeline.pipeline_id)

        # Use query_id=1 from seed data instead of string
        result = await pipeline._generate(1, top_k=4)

        assert result.metadata is not None
        assert len(result.metadata.get("chunk_ids", [])) <= 5

    @pytest.mark.asyncio
    async def test_aggregates_token_usage(self, session_factory, cleanup):
        """Test token usage is aggregated across all LLM calls."""
        from autorag_research.pipelines.generation.ircot import IRCoTGenerationPipeline

        call_count = [0]
        tokens_per_call = 150

        async def mock_ainvoke(prompt):
            call_count[0] += 1
            response = MagicMock()
            response.content = "The answer is: done." if call_count[0] == 2 else "Thinking."
            response.usage_metadata = {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": tokens_per_call,
            }
            return response

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)

        pipeline = IRCoTGenerationPipeline(
            session_factory=session_factory,
            name="test_tokens",
            llm=mock_llm,
            retrieval_pipeline=create_mock_retrieval_pipeline(),
            max_steps=4,
        )
        cleanup.append(pipeline.pipeline_id)

        # Use query_id=1 from seed data instead of string
        result = await pipeline._generate(1, top_k=4)

        assert result.token_usage is not None
        assert result.token_usage["total_tokens"] == call_count[0] * tokens_per_call

    @pytest.mark.asyncio
    async def test_case_insensitive_termination(self, session_factory, cleanup):
        """Test termination detection is case-insensitive."""
        from autorag_research.pipelines.generation.ircot import IRCoTGenerationPipeline

        mock_llm = create_mock_llm()
        call_count = [0]
        responses = ["The ANSWER IS: Test.", "Final."]

        async def mock_ainvoke(prompt):
            response = MagicMock()
            response.content = responses[min(call_count[0], len(responses) - 1)]
            response.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
            call_count[0] += 1
            return response

        mock_llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
        mock_retrieval = create_mock_retrieval_pipeline()

        pipeline = IRCoTGenerationPipeline(
            session_factory=session_factory,
            name="test_case",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval,
            max_steps=5,
        )
        cleanup.append(pipeline.pipeline_id)

        # Use query_id=2 from seed data instead of string
        result = await pipeline._generate(2, top_k=2)

        # Verify early termination (step 1, not 5)
        assert result.metadata is not None
        assert result.metadata.get("steps") == 1
        assert result.text is not None


# ==================== Integration Tests with PipelineTestVerifier ====================


class TestIRCoTPipelineIntegration:
    """Integration tests for IRCoT pipeline using PipelineTestVerifier."""

    def test_full_flow(self, session_factory, cleanup):
        """Test end-to-end IRCoT flow with PipelineTestVerifier."""
        from autorag_research.pipelines.generation.ircot import IRCoTGenerationPipeline

        responses = []
        for _ in range(20):
            responses.extend(["Reasoning.", "The answer is: done.", "Final."])

        pipeline = IRCoTGenerationPipeline(
            session_factory=session_factory,
            name="test_full_flow",
            llm=FakeListLLM(responses=responses),
            retrieval_pipeline=create_mock_retrieval_pipeline(),
            k_per_step=3,
            max_steps=4,
        )
        cleanup.append(pipeline.pipeline_id)

        result = pipeline.run(top_k=3, batch_size=10)

        # Use PipelineTestVerifier for comprehensive validation
        config = PipelineTestConfig(
            pipeline_type="generation",
            expected_total_queries=5,  # Seed data has 5 queries
            check_token_usage=False,  # FakeListLLM doesn't provide token usage
            check_execution_time=True,
            check_persistence=True,
        )
        verifier = PipelineTestVerifier(result, pipeline.pipeline_id, session_factory, config)
        verifier.verify_all()

    @pytest.mark.asyncio
    async def test_handles_empty_retrieval(self, session_factory, cleanup):
        """Test IRCoT handles empty retrieval results gracefully."""
        from autorag_research.pipelines.generation.ircot import IRCoTGenerationPipeline

        mock_llm = create_mock_llm()
        call_count = [0]
        responses = ["Reasoning.", "The answer is: none.", "None."]

        async def mock_ainvoke(prompt):
            response = MagicMock()
            response.content = responses[min(call_count[0], len(responses) - 1)]
            response.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
            call_count[0] += 1
            return response

        mock_llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)

        pipeline = IRCoTGenerationPipeline(
            session_factory=session_factory,
            name="test_empty",
            llm=mock_llm,
            retrieval_pipeline=create_mock_retrieval_pipeline(default_results=[]),
            max_steps=2,
        )
        cleanup.append(pipeline.pipeline_id)

        # Use query_id=1 from seed data instead of string
        result = await pipeline._generate(1, top_k=5)

        # Verify generation completes without error
        assert result.text is not None
        assert result.metadata is not None

    @pytest.mark.asyncio
    async def test_metadata_contains_cot_history(self, session_factory, cleanup):
        """Test result metadata includes chain-of-thought history."""
        from autorag_research.pipelines.generation.ircot import IRCoTGenerationPipeline

        mock_llm = create_mock_llm()
        call_count = [0]
        responses = ["Step 1.", "The answer is: found.", "Final."]

        async def mock_ainvoke(prompt):
            response = MagicMock()
            response.content = responses[min(call_count[0], len(responses) - 1)]
            response.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
            call_count[0] += 1
            return response

        mock_llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)

        pipeline = IRCoTGenerationPipeline(
            session_factory=session_factory,
            name="test_cot_meta",
            llm=mock_llm,
            retrieval_pipeline=create_mock_retrieval_pipeline(),
            max_steps=4,
        )
        cleanup.append(pipeline.pipeline_id)

        # Use query_id=1 from seed data instead of string
        result = await pipeline._generate(1, top_k=3)

        assert result.metadata is not None
        assert "cot_sentences" in result.metadata

    @pytest.mark.asyncio
    async def test_metadata_contains_chunk_ids(self, session_factory, cleanup):
        """Test result metadata includes retrieved chunk IDs."""
        from autorag_research.pipelines.generation.ircot import IRCoTGenerationPipeline

        mock_llm = create_mock_llm()
        call_count = [0]
        responses = ["The answer is: quick.", "Final."]

        async def mock_ainvoke(prompt):
            response = MagicMock()
            response.content = responses[min(call_count[0], len(responses) - 1)]
            response.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
            call_count[0] += 1
            return response

        mock_llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)

        pipeline = IRCoTGenerationPipeline(
            session_factory=session_factory,
            name="test_chunk_meta",
            llm=mock_llm,
            retrieval_pipeline=create_mock_retrieval_pipeline(),
            max_steps=2,
        )
        cleanup.append(pipeline.pipeline_id)

        # Use query_id=1 from seed data instead of string
        result = await pipeline._generate(1, top_k=3)

        assert result.metadata is not None
        assert "chunk_ids" in result.metadata

    @pytest.mark.asyncio
    async def test_custom_prompt_templates(self, session_factory, cleanup):
        """Test IRCoT with custom prompt templates."""
        from autorag_research.pipelines.generation.ircot import IRCoTGenerationPipeline

        captured_prompts = []

        async def mock_ainvoke(prompt):
            captured_prompts.append(prompt)
            response = MagicMock()
            response.content = "The answer is: done." if len(captured_prompts) == 1 else "Final."
            response.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
            return response

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)

        pipeline = IRCoTGenerationPipeline(
            session_factory=session_factory,
            name="test_templates",
            llm=mock_llm,
            retrieval_pipeline=create_mock_retrieval_pipeline(),
            reasoning_prompt_template="CUSTOM_REASONING: {query}\n{paragraphs}\n{cot_history}",
            qa_prompt_template="CUSTOM_QA: {query}\n{paragraphs}",
        )
        cleanup.append(pipeline.pipeline_id)

        # Use query_id=2 from seed data instead of string
        await pipeline._generate(2, top_k=2)

        assert "CUSTOM_REASONING:" in captured_prompts[0]
        assert "CUSTOM_QA:" in captured_prompts[1]

    @pytest.mark.asyncio
    async def test_extracts_first_sentence_only(self, session_factory, cleanup):
        """Test only first sentence is extracted for CoT history."""
        from autorag_research.pipelines.generation.ircot import IRCoTGenerationPipeline

        mock_llm = create_mock_llm()
        call_count = [0]
        responses = [
            "First. Second. Third.",
            "The answer is: found. More.",
            "Final.",
        ]

        async def mock_ainvoke(prompt):
            response = MagicMock()
            response.content = responses[min(call_count[0], len(responses) - 1)]
            response.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
            call_count[0] += 1
            return response

        mock_llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)

        pipeline = IRCoTGenerationPipeline(
            session_factory=session_factory,
            name="test_first_sent",
            llm=mock_llm,
            retrieval_pipeline=create_mock_retrieval_pipeline(),
            max_steps=3,
        )
        cleanup.append(pipeline.pipeline_id)

        # Use query_id=1 from seed data instead of string
        result = await pipeline._generate(1, top_k=2)

        cot = result.metadata.get("cot_sentences", [])
        assert len(cot) > 0
        for sentence in cot:
            parts = sentence.split(". ")
            assert len(parts) <= 2
            if len(parts) == 2:
                assert parts[1] == ""

    def test_full_run_with_token_usage(self, session_factory, cleanup):
        """Test full pipeline run with token usage tracking."""
        from autorag_research.pipelines.generation.ircot import IRCoTGenerationPipeline

        call_count = [0]

        async def mock_ainvoke(prompt):
            call_count[0] += 1
            response = MagicMock()
            response.content = "The answer is: done." if call_count[0] % 2 == 1 else "Final."
            response.usage_metadata = {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            }
            return response

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)

        pipeline = IRCoTGenerationPipeline(
            session_factory=session_factory,
            name="test_full_tokens",
            llm=mock_llm,
            retrieval_pipeline=create_mock_retrieval_pipeline(),
            max_steps=2,
        )
        cleanup.append(pipeline.pipeline_id)

        result = pipeline.run(top_k=3, batch_size=10)

        # Use PipelineTestVerifier with token usage validation
        config = PipelineTestConfig(
            pipeline_type="generation",
            expected_total_queries=5,
            check_token_usage=True,
            check_execution_time=True,
            check_persistence=True,
            expected_token_usage_keys=["prompt_tokens", "completion_tokens", "total_tokens"],
        )
        verifier = PipelineTestVerifier(result, pipeline.pipeline_id, session_factory, config)
        verifier.verify_all()
