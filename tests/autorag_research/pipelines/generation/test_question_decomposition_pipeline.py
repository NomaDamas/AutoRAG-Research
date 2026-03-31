"""Tests for the Question Decomposition generation pipeline.

Test Categories:
1. Unit Tests - Helper methods for parsing, prompt building, and merging
2. Pipeline Initialization Tests - Constructor and config
3. Core Algorithm Tests - Decomposition, retrieval merging, token aggregation
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


@pytest.fixture
def cleanup(session_factory):
    """Cleanup fixture for pipeline results."""
    yield from cleanup_pipeline_results_factory(session_factory)


@pytest.fixture
def mock_retrieval():
    """Create a mock retrieval pipeline."""
    return create_mock_retrieval_pipeline()


@pytest.fixture
def question_decomposition_pipeline(session_factory, cleanup):
    """Create a Question Decomposition pipeline with default settings."""
    from autorag_research.pipelines.generation.question_decomposition import (
        QuestionDecompositionPipeline,
    )

    pipeline = QuestionDecompositionPipeline(
        session_factory=session_factory,
        name="test_question_decomposition_fixture",
        llm=create_mock_llm(),
        retrieval_pipeline=create_mock_retrieval_pipeline(),
    )
    cleanup.append(pipeline.pipeline_id)
    return pipeline


class TestParseSubQuestions:
    """Unit tests for _parse_subquestions helper method."""

    @pytest.fixture
    def pipeline_class(self):
        from autorag_research.pipelines.generation.question_decomposition import (
            QuestionDecompositionPipeline,
        )

        return QuestionDecompositionPipeline

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("1. What is A?\n2. What is B?", ["What is A?", "What is B?"]),
            ("- Find the author\n- Find the date", ["Find the author", "Find the date"]),
            ("What is A? What is B?", ["What is A?", "What is B?"]),
            ("Single sub-question", ["Single sub-question"]),
            ("", []),
        ],
        ids=["numbered", "bullet", "inline_questions", "single", "empty"],
    )
    def test_parse_subquestions(self, pipeline_class, text, expected):
        """Test sub-question parsing with different output formats."""
        result = pipeline_class._parse_subquestions(None, text)
        assert result == expected


class TestPromptBuilders:
    """Unit tests for prompt building methods."""

    def test_build_decomposition_prompt(self, question_decomposition_pipeline):
        """Test decomposition prompt contains query and configured limit."""
        prompt = question_decomposition_pipeline._build_decomposition_prompt("What is A?")
        assert "What is A?" in prompt
        assert str(question_decomposition_pipeline.max_subquestions) in prompt

    def test_build_qa_prompt(self, question_decomposition_pipeline):
        """Test QA prompt is built correctly."""
        prompt = question_decomposition_pipeline._build_qa_prompt("Question?", ["Doc 1", "Doc 2"])
        assert "Question?" in prompt
        assert "[1]" in prompt and "[2]" in prompt
        assert "Doc 1" in prompt and "Doc 2" in prompt


class TestMergeResults:
    """Unit tests for retrieval merge behavior."""

    def test_merge_results_keeps_highest_score(self, question_decomposition_pipeline):
        """Test duplicate doc_ids keep the highest observed score."""
        merged = question_decomposition_pipeline._merge_results([
            [{"doc_id": 1, "score": 0.4}, {"doc_id": 2, "score": 0.7}],
            [{"doc_id": 1, "score": 0.9}, {"doc_id": 3, "score": 0.6}],
        ])

        assert merged == [
            {"doc_id": 1, "score": 0.9},
            {"doc_id": 2, "score": 0.7},
            {"doc_id": 3, "score": 0.6},
        ]


class TestQuestionDecompositionPipelineInitialization:
    """Tests for Question Decomposition pipeline initialization and config."""

    def test_initialization_stores_config(self, session_factory, mock_retrieval, cleanup):
        """Test pipeline stores custom parameters correctly."""
        from autorag_research.pipelines.generation.question_decomposition import (
            QuestionDecompositionPipeline,
        )

        pipeline = QuestionDecompositionPipeline(
            session_factory=session_factory,
            name="test_question_decomposition_init",
            llm=create_mock_llm(),
            retrieval_pipeline=mock_retrieval,
            decomposition_prompt_template="DECOMP: {query} ({max_subquestions})",
            qa_prompt_template="QA: {query}\n{paragraphs}",
            max_subquestions=5,
        )
        cleanup.append(pipeline.pipeline_id)

        assert pipeline.pipeline_id > 0
        assert pipeline.max_subquestions == 5
        assert pipeline._decomposition_prompt_template == "DECOMP: {query} ({max_subquestions})"
        assert pipeline._qa_prompt_template == "QA: {query}\n{paragraphs}"

    def test_initialization_default_values(self, session_factory, mock_retrieval, cleanup):
        """Test pipeline uses correct default values."""
        from autorag_research.pipelines.generation.question_decomposition import (
            QuestionDecompositionPipeline,
        )

        pipeline = QuestionDecompositionPipeline(
            session_factory=session_factory,
            name="test_question_decomposition_defaults",
            llm=create_mock_llm(),
            retrieval_pipeline=mock_retrieval,
        )
        cleanup.append(pipeline.pipeline_id)

        assert pipeline.max_subquestions == 3

    def test_get_pipeline_config(self, session_factory, mock_retrieval, cleanup):
        """Test _get_pipeline_config returns correct configuration."""
        from autorag_research.pipelines.generation.question_decomposition import (
            QuestionDecompositionPipeline,
        )

        pipeline = QuestionDecompositionPipeline(
            session_factory=session_factory,
            name="test_question_decomposition_config",
            llm=create_mock_llm(),
            retrieval_pipeline=mock_retrieval,
            max_subquestions=4,
        )
        cleanup.append(pipeline.pipeline_id)

        config = pipeline._get_pipeline_config()
        assert config["type"] == "question_decomposition"
        assert config["max_subquestions"] == 4
        assert config["retrieval_pipeline_id"] == mock_retrieval.pipeline_id
        assert "decomposition_prompt_template" in config
        assert "qa_prompt_template" in config


class TestQuestionDecompositionAlgorithm:
    """Tests for Question Decomposition core algorithm behavior."""

    @pytest.mark.asyncio
    async def test_retrieves_original_and_subquestions(self, session_factory, cleanup):
        """Test retrieval runs for the original query and each unique sub-question."""
        from autorag_research.pipelines.generation.question_decomposition import (
            QuestionDecompositionPipeline,
        )

        captured_prompts = []

        async def mock_ainvoke(prompt):
            captured_prompts.append(prompt)
            response = MagicMock()
            response.content = (
                "1. Who is Doc One about?\n2. When was Doc One written?"
                if len(captured_prompts) == 1
                else "Final answer."
            )
            response.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
            return response

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)

        mock_retrieval = MagicMock()
        mock_retrieval.pipeline_id = 1
        mock_retrieval._retrieve_by_id = AsyncMock(
            return_value=[{"doc_id": 1, "score": 0.7}, {"doc_id": 2, "score": 0.6}]
        )
        mock_retrieval.retrieve = AsyncMock(
            side_effect=[
                [{"doc_id": 3, "score": 0.95}],
                [{"doc_id": 4, "score": 0.8}],
            ]
        )

        pipeline = QuestionDecompositionPipeline(
            session_factory=session_factory,
            name="test_question_decomposition_retrievals",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval,
        )
        cleanup.append(pipeline.pipeline_id)

        result = await pipeline._generate(1, top_k=2)

        mock_retrieval._retrieve_by_id.assert_awaited_once_with(1, 2)
        assert mock_retrieval.retrieve.await_count == 2
        assert result.metadata is not None
        assert result.metadata["sub_questions"] == [
            "Who is Doc One about?",
            "When was Doc One written?",
        ]
        assert result.metadata["retrieved_chunk_ids"] == [3, 4]

    @pytest.mark.asyncio
    async def test_deduplicates_docs_by_highest_score_and_applies_top_k(self, session_factory, cleanup):
        """Test merged retrieval keeps max score per doc and trims to final top-k."""
        from autorag_research.pipelines.generation.question_decomposition import (
            QuestionDecompositionPipeline,
        )

        call_count = [0]

        async def mock_ainvoke(prompt):
            response = MagicMock()
            response.content = (
                "1. Who wrote Doc One?\n2. What is the publication date?" if call_count[0] == 0 else "Final answer."
            )
            response.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
            call_count[0] += 1
            return response

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)

        mock_retrieval = MagicMock()
        mock_retrieval.pipeline_id = 1
        mock_retrieval._retrieve_by_id = AsyncMock(
            return_value=[{"doc_id": 1, "score": 0.4}, {"doc_id": 2, "score": 0.9}]
        )
        mock_retrieval.retrieve = AsyncMock(
            side_effect=[
                [{"doc_id": 1, "score": 0.95}, {"doc_id": 3, "score": 0.5}],
                [{"doc_id": 2, "score": 0.85}, {"doc_id": 4, "score": 0.8}],
            ]
        )

        pipeline = QuestionDecompositionPipeline(
            session_factory=session_factory,
            name="test_question_decomposition_dedup",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval,
        )
        cleanup.append(pipeline.pipeline_id)

        result = await pipeline._generate(1, top_k=3)

        assert result.metadata is not None
        assert result.metadata["retrieved_chunk_ids"] == [1, 2, 4]
        assert result.metadata["retrieved_scores"] == [0.95, 0.9, 0.8]

    @pytest.mark.asyncio
    async def test_aggregates_token_usage(self, session_factory, cleanup):
        """Test token usage is aggregated across decomposition and answer generation."""
        from autorag_research.pipelines.generation.question_decomposition import (
            QuestionDecompositionPipeline,
        )

        call_count = [0]

        async def mock_ainvoke(prompt):
            call_count[0] += 1
            response = MagicMock()
            response.content = "1. Sub-question?" if call_count[0] == 1 else "Final answer."
            response.usage_metadata = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
            return response

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)

        pipeline = QuestionDecompositionPipeline(
            session_factory=session_factory,
            name="test_question_decomposition_tokens",
            llm=mock_llm,
            retrieval_pipeline=create_mock_retrieval_pipeline(),
        )
        cleanup.append(pipeline.pipeline_id)

        result = await pipeline._generate(1, top_k=3)

        assert result.token_usage is not None
        assert result.token_usage["total_tokens"] == 300
        assert result.token_usage["prompt_tokens"] == 200
        assert result.token_usage["completion_tokens"] == 100

    @pytest.mark.asyncio
    async def test_custom_prompt_templates(self, session_factory, cleanup):
        """Test custom prompt templates are used for both LLM calls."""
        from autorag_research.pipelines.generation.question_decomposition import (
            QuestionDecompositionPipeline,
        )

        captured_prompts = []

        async def mock_ainvoke(prompt):
            captured_prompts.append(prompt)
            response = MagicMock()
            response.content = "1. Sub-question?" if len(captured_prompts) == 1 else "Final answer."
            response.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
            return response

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)

        pipeline = QuestionDecompositionPipeline(
            session_factory=session_factory,
            name="test_question_decomposition_templates",
            llm=mock_llm,
            retrieval_pipeline=create_mock_retrieval_pipeline(),
            decomposition_prompt_template="CUSTOM_DECOMP: {query} ({max_subquestions})",
            qa_prompt_template="CUSTOM_QA: {query}\n{paragraphs}",
        )
        cleanup.append(pipeline.pipeline_id)

        await pipeline._generate(2, top_k=2)

        assert "CUSTOM_DECOMP:" in captured_prompts[0]
        assert "CUSTOM_QA:" in captured_prompts[1]

    @pytest.mark.asyncio
    async def test_filters_duplicate_subquestions_and_original_query(self, session_factory, cleanup):
        """Test sub-question list excludes duplicates and the original query."""
        from autorag_research.pipelines.generation.question_decomposition import (
            QuestionDecompositionPipeline,
        )

        mock_llm = MagicMock()
        responses = [
            "What is Doc One about?\nWhat is Doc One about?\nWho wrote Doc One?\nWho wrote Doc One?",
            "Final answer.",
        ]
        call_count = [0]

        async def mock_ainvoke(prompt):
            response = MagicMock()
            response.content = responses[min(call_count[0], len(responses) - 1)]
            response.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
            call_count[0] += 1
            return response

        mock_llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)

        mock_retrieval = create_mock_retrieval_pipeline()
        pipeline = QuestionDecompositionPipeline(
            session_factory=session_factory,
            name="test_question_decomposition_dedup_subquestions",
            llm=mock_llm,
            retrieval_pipeline=mock_retrieval,
        )
        cleanup.append(pipeline.pipeline_id)

        result = await pipeline._generate(1, top_k=3)

        assert result.metadata is not None
        assert result.metadata["sub_questions"] == ["Who wrote Doc One?"]
        assert mock_retrieval.retrieve.await_count == 1

    @pytest.mark.asyncio
    async def test_handles_empty_retrieval(self, session_factory, cleanup):
        """Test Question Decomposition handles empty retrieval results gracefully."""
        from autorag_research.pipelines.generation.question_decomposition import (
            QuestionDecompositionPipeline,
        )

        mock_llm = MagicMock()
        call_count = [0]

        async def mock_ainvoke(prompt):
            response = MagicMock()
            response.content = "1. Missing evidence?" if call_count[0] == 0 else "Final answer."
            response.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
            call_count[0] += 1
            return response

        mock_llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)

        pipeline = QuestionDecompositionPipeline(
            session_factory=session_factory,
            name="test_question_decomposition_empty",
            llm=mock_llm,
            retrieval_pipeline=create_mock_retrieval_pipeline(default_results=[]),
        )
        cleanup.append(pipeline.pipeline_id)

        result = await pipeline._generate(1, top_k=5)

        assert result.text is not None
        assert result.metadata is not None
        assert result.metadata["retrieved_chunk_ids"] == []


class TestQuestionDecompositionPipelineIntegration:
    """Integration tests for the Question Decomposition pipeline."""

    def test_full_flow(self, session_factory, cleanup):
        """Test end-to-end Question Decomposition flow with PipelineTestVerifier."""
        from autorag_research.pipelines.generation.question_decomposition import (
            QuestionDecompositionPipeline,
        )

        responses = []
        for _ in range(10):
            responses.extend([
                "1. What evidence is relevant?\n2. What supporting detail is needed?",
                "Final answer.",
            ])

        pipeline = QuestionDecompositionPipeline(
            session_factory=session_factory,
            name="test_question_decomposition_full_flow",
            llm=FakeListLLM(responses=responses),
            retrieval_pipeline=create_mock_retrieval_pipeline(),
            max_subquestions=2,
        )
        cleanup.append(pipeline.pipeline_id)

        result = pipeline.run(top_k=3, batch_size=10)

        config = PipelineTestConfig(
            pipeline_type="generation",
            expected_total_queries=5,
            check_token_usage=False,
            check_execution_time=True,
            check_persistence=True,
        )
        verifier = PipelineTestVerifier(result, pipeline.pipeline_id, session_factory, config)
        verifier.verify_all()
