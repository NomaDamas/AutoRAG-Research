"""Tests for the RAG-Critic generation pipeline."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.autorag_research.pipelines.pipeline_test_utils import (
    PipelineTestConfig,
    PipelineTestVerifier,
    cleanup_pipeline_results_factory,
    create_mock_retrieval_pipeline,
)


@pytest.fixture
def cleanup(session_factory):
    """Cleanup fixture for pipeline results."""
    yield from cleanup_pipeline_results_factory(session_factory)


def create_lightweight_rag_critic_pipeline(
    *,
    llm: MagicMock | None = None,
    retrieval_pipeline: MagicMock | None = None,
    max_iterations: int = 2,
    max_actions_per_iteration: int = 4,
):
    """Create a RAG-Critic pipeline instance without DB-backed persistence."""
    from autorag_research.pipelines.generation.rag_critic import (
        DEFAULT_ANSWER_PROMPT,
        DEFAULT_CRITIC_PROMPT,
        DEFAULT_DECOMPOSITION_PROMPT,
        DEFAULT_PLANNER_PROMPT,
        DEFAULT_REFINE_PROMPT,
        DEFAULT_REWRITE_PROMPT,
        RAGCriticPipeline,
    )

    pipeline = object.__new__(RAGCriticPipeline)
    pipeline._answer_prompt_template = DEFAULT_ANSWER_PROMPT
    pipeline._critic_prompt_template = DEFAULT_CRITIC_PROMPT
    pipeline._planner_prompt_template = DEFAULT_PLANNER_PROMPT
    pipeline._rewrite_prompt_template = DEFAULT_REWRITE_PROMPT
    pipeline._decomposition_prompt_template = DEFAULT_DECOMPOSITION_PROMPT
    pipeline._refine_prompt_template = DEFAULT_REFINE_PROMPT
    pipeline._max_iterations = max_iterations
    pipeline._max_actions_per_iteration = max_actions_per_iteration
    pipeline._llm = llm or MagicMock()
    pipeline._retrieval_pipeline = retrieval_pipeline or create_mock_retrieval_pipeline()
    pipeline._service = MagicMock()
    pipeline._service.get_query_text.return_value = "What is RAG-Critic?"
    pipeline._service.get_chunk_contents.side_effect = lambda chunk_ids: [f"chunk {chunk_id}" for chunk_id in chunk_ids]
    pipeline.pipeline_id = 999
    return pipeline


class TestRAGCriticPipelineUnit:
    """Unit tests for RAG-Critic pipeline helpers and configuration."""

    def test_parse_json_payload_handles_code_fence(self):
        """Structured outputs should parse even when wrapped in markdown fences."""
        from autorag_research.pipelines.generation.rag_critic import RAGCriticPipeline

        payload = RAGCriticPipeline._parse_json_payload(
            """```json
            {"verdict": "revise", "recommended_actions": ["retrieval"]}
            ```"""
        )

        assert payload == {
            "verdict": "revise",
            "recommended_actions": ["retrieval"],
        }

    def test_parse_json_payload_handles_top_level_array(self):
        """Structured outputs may also be top-level arrays."""
        from autorag_research.pipelines.generation.rag_critic import RAGCriticPipeline

        payload = RAGCriticPipeline._parse_json_payload(
            """```json
            [{"action": "retrieval"}, {"action": "generate_answer"}]
            ```"""
        )

        assert payload == [
            {"action": "retrieval"},
            {"action": "generate_answer"},
        ]

    def test_pipeline_config_contains_required_fields(self):
        """Pipeline config should expose the critic-specific knobs without DB setup."""
        retrieval_pipeline = create_mock_retrieval_pipeline()
        pipeline = create_lightweight_rag_critic_pipeline(
            retrieval_pipeline=retrieval_pipeline,
            max_iterations=3,
            max_actions_per_iteration=4,
        )

        config = pipeline._get_pipeline_config()

        assert config["type"] == "rag_critic"
        assert config["max_iterations"] == 3
        assert config["max_actions_per_iteration"] == 4
        assert config["retrieval_pipeline_id"] == retrieval_pipeline.pipeline_id
        assert "answer_prompt_template" in config
        assert "critic_prompt_template" in config
        assert "planner_prompt_template" in config

    @pytest.mark.asyncio
    async def test_plan_actions_accepts_top_level_array_payload(self):
        """Planner responses may be a top-level array of action objects."""
        from autorag_research.util import TokenUsageTracker

        pipeline = create_lightweight_rag_critic_pipeline()

        pipeline._invoke_and_record = AsyncMock(  # type: ignore[method-assign]
            return_value='[{"action": "retrieval"}, {"action": "generate_answer"}]'
        )

        actions = await pipeline._plan_actions(
            "What is RAG-Critic?",
            "Initial answer",
            {"feedback": "Need better grounding", "recommended_actions": ["retrieval"]},
            TokenUsageTracker(),
        )

        assert actions == [
            {"action": "retrieval"},
            {"action": "generate_answer"},
        ]

    @pytest.mark.asyncio
    async def test_decompose_query_accepts_top_level_array_payload(self):
        """Decomposition responses may be a top-level array of sub-question strings."""
        from autorag_research.util import TokenUsageTracker

        pipeline = create_lightweight_rag_critic_pipeline()

        pipeline._invoke_and_record = AsyncMock(  # type: ignore[method-assign]
            return_value='["What is RAG?", "How does the critic refine it?"]'
        )

        sub_questions = await pipeline._decompose_query(
            "Explain RAG-Critic",
            "Need smaller retrieval queries",
            "Split the problem",
            TokenUsageTracker(),
        )

        assert sub_questions == [
            "What is RAG?",
            "How does the critic refine it?",
        ]

    @pytest.mark.asyncio
    async def test_generate_executes_critic_plan_actions(self):
        """A revise verdict should trigger planner actions before approval."""
        prompt_log: list[str] = []

        async def mock_ainvoke(prompt: str):
            prompt_log.append(prompt)
            response = MagicMock()
            if "Return only a rewritten query" in prompt:
                response.content = "rewritten founder query"
            elif "Return JSON with a sub_questions array" in prompt:
                response.content = '{"sub_questions": ["When was it founded?", "Who founded it?"]}'
            elif "Return only the refined document" in prompt:
                response.content = "Refined supporting evidence"
            elif "critic for a retrieval-augmented generation" in prompt and "Current answer:" in prompt:
                if "Improved grounded answer" in prompt:
                    response.content = '{"verdict": "approved", "feedback": "grounded"}'
                else:
                    response.content = (
                        '{"verdict": "revise", "feedback": "Need stronger evidence", '
                        '"recommended_actions": ["rewrite_query", "retrieval", "decompose_query", '
                        '"retrieval", "refine_documents", "generate_answer"]}'
                    )
            elif "critic-guided planner" in prompt:
                response.content = (
                    '{"actions": ['
                    '{"action": "rewrite_query", "instruction": "focus on the founder"}, '
                    '{"action": "retrieval", "query_source": "rewritten_query", "top_k": 2, "strategy": "replace"}, '
                    '{"action": "decompose_query", "instruction": "split into founding sub-questions"}, '
                    '{"action": "retrieval", "query_source": "sub_questions", "top_k": 1, "strategy": "append"}, '
                    '{"action": "refine_documents", "instruction": "keep only founder facts"}, '
                    '{"action": "generate_answer", "instruction": "cite the founder evidence clearly"}'
                    "]}"
                )
            else:
                response.content = (
                    "Improved grounded answer"
                    if "cite the founder evidence clearly" in prompt
                    else "Initial draft answer"
                )

            response.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
            response.response_metadata = {}
            return response

        llm = MagicMock()
        llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)

        retrieval_pipeline = create_mock_retrieval_pipeline()
        retrieval_pipeline._retrieve_by_id = AsyncMock(
            return_value=[{"doc_id": 1, "score": 0.35}, {"doc_id": 2, "score": 0.3}]
        )
        retrieval_pipeline.retrieve = AsyncMock(
            side_effect=[
                [{"doc_id": 3, "score": 0.92}, {"doc_id": 4, "score": 0.88}],
                [{"doc_id": 5, "score": 0.91}],
                [{"doc_id": 6, "score": 0.87}],
            ]
        )

        pipeline = create_lightweight_rag_critic_pipeline(
            llm=llm,
            retrieval_pipeline=retrieval_pipeline,
            max_iterations=2,
            max_actions_per_iteration=6,
        )

        result = await pipeline._generate(1, top_k=2)

        assert result.text == "Improved grounded answer"
        assert result.metadata is not None
        assert result.metadata["iteration_count"] == 2
        assert result.metadata["rewritten_queries"] == ["rewritten founder query"]
        assert result.metadata["sub_questions"] == ["When was it founded?", "Who founded it?"]
        assert [step["action"] for step in result.metadata["executed_actions"]] == [
            "rewrite_query",
            "retrieval",
            "decompose_query",
            "retrieval",
            "refine_documents",
            "generate_answer",
        ]
        assert result.metadata["retrieved_chunk_ids"] == [3, 4, 5, 6]
        assert result.token_usage == {"prompt_tokens": 80, "completion_tokens": 40, "total_tokens": 120}
        retrieval_pipeline._retrieve_by_id.assert_awaited_once_with(1, 2)
        assert retrieval_pipeline.retrieve.await_count == 3
        assert any("Need stronger evidence" in critique["feedback"] for critique in result.metadata["critique_history"])
        assert any("critic-guided planner" in prompt for prompt in prompt_log)

    @pytest.mark.asyncio
    async def test_retrieval_without_query_source_uses_working_query(self):
        """Planner retrieval should use the rewritten working query when query_source is omitted."""

        async def mock_ainvoke(prompt: str):
            response = MagicMock()
            if "Return only a rewritten query" in prompt:
                response.content = "rewritten working query"
            elif "critic for a retrieval-augmented generation" in prompt and "Current answer:" in prompt:
                if "Rewritten-query answer" in prompt:
                    response.content = '{"verdict": "approved", "feedback": "grounded"}'
                else:
                    response.content = (
                        '{"verdict": "revise", "feedback": "Use the improved query", '
                        '"recommended_actions": ["rewrite_query", "retrieval", "generate_answer"]}'
                    )
            elif "critic-guided planner" in prompt:
                response.content = (
                    '{"actions": ['
                    '{"action": "rewrite_query", "instruction": "focus on the key entity"}, '
                    '{"action": "retrieval", "top_k": 1, "strategy": "replace"}, '
                    '{"action": "generate_answer", "instruction": "answer using the refreshed evidence"}'
                    "]}"
                )
            else:
                response.content = (
                    "Rewritten-query answer" if "answer using the refreshed evidence" in prompt else "Initial answer"
                )

            response.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
            response.response_metadata = {}
            return response

        llm = MagicMock()
        llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)

        retrieval_pipeline = create_mock_retrieval_pipeline()
        retrieval_pipeline._retrieve_by_id = AsyncMock(return_value=[{"doc_id": 1, "score": 0.4}])
        retrieval_pipeline.retrieve = AsyncMock(return_value=[{"doc_id": 3, "score": 0.91}])

        pipeline = create_lightweight_rag_critic_pipeline(
            llm=llm,
            retrieval_pipeline=retrieval_pipeline,
            max_iterations=2,
            max_actions_per_iteration=3,
        )

        result = await pipeline._generate(1, top_k=1)

        assert result.text == "Rewritten-query answer"
        retrieval_pipeline.retrieve.assert_awaited_once_with("rewritten working query", 1)


class TestRAGCriticPipelineIntegration:
    """Integration tests for running RAG-Critic through the service layer."""

    @staticmethod
    def _create_multi_query_llm() -> MagicMock:
        """Create a mock LLM that cycles through answer -> critic -> planner -> answer -> critic."""
        call_count = [0]

        async def mock_ainvoke(prompt: str):
            response = MagicMock()
            phase = call_count[0] % 5
            call_count[0] += 1

            if phase == 0:
                response.content = "Initial draft answer"
            elif phase == 1:
                response.content = '{"verdict": "revise", "feedback": "missing evidence", "recommended_actions": ["retrieval", "generate_answer"]}'
            elif phase == 2:
                response.content = '{"actions": [{"action": "retrieval", "query_source": "original", "top_k": 2, "strategy": "append"}, {"action": "generate_answer", "instruction": "ground the answer with the retrieved documents"}]}'
            elif phase == 3:
                response.content = "Grounded final answer"
            else:
                response.content = '{"verdict": "approved", "feedback": "looks good"}'

            response.usage_metadata = {"input_tokens": 8, "output_tokens": 4, "total_tokens": 12}
            response.response_metadata = {}
            return response

        mock = MagicMock()
        mock.ainvoke = AsyncMock(side_effect=mock_ainvoke)
        return mock

    def test_run_pipeline_with_verifier(self, session_factory, cleanup):
        """The pipeline should persist results and aggregate token usage across queries."""
        from autorag_research.pipelines.generation.rag_critic import RAGCriticPipeline

        retrieval_pipeline = create_mock_retrieval_pipeline()
        pipeline = RAGCriticPipeline(
            session_factory=session_factory,
            name="test_rag_critic_integration",
            llm=self._create_multi_query_llm(),
            retrieval_pipeline=retrieval_pipeline,
            max_iterations=2,
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
        assert result["token_usage"] == {
            "prompt_tokens": 200,
            "completion_tokens": 100,
            "total_tokens": 300,
        }
        assert retrieval_pipeline.retrieve.await_count == 5
