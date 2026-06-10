from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from langchain_core.language_models.fake import FakeListLLM

from autorag_research.pipelines.generation.adaptive_rag import AdaptiveRAGPipeline, AdaptiveRAGPipelineConfig
from tests.autorag_research.pipelines.pipeline_test_utils import (
    PipelineTestConfig,
    PipelineTestVerifier,
    cleanup_pipeline_results_factory,
    create_mock_llm,
    create_mock_retrieval_pipeline,
)


@pytest.fixture
def cleanup(session_factory):
    yield from cleanup_pipeline_results_factory(session_factory)


@pytest.fixture
def mock_retrieval():
    return create_mock_retrieval_pipeline(
        default_results=[
            {"doc_id": 1, "score": 0.95},
            {"doc_id": 2, "score": 0.85},
            {"doc_id": 3, "score": 0.75},
        ]
    )


class TestAdaptiveRAGPipeline:
    @pytest.mark.asyncio
    async def test_zero_route_skips_retrieval(self, session_factory, mock_retrieval, cleanup):
        llm = FakeListLLM(responses=["simple", "Direct answer"])
        pipeline = AdaptiveRAGPipeline(
            session_factory=session_factory,
            name="test_adaptive_rag_zero",
            llm=llm,
            retrieval_pipeline=mock_retrieval,
        )
        cleanup.append(pipeline.pipeline_id)

        result = await pipeline._generate(query_id=1, top_k=3)

        assert result.text == "Direct answer"
        assert result.metadata is not None
        assert result.metadata["complexity_tier"] == "simple"
        assert result.metadata["route"] == "zero"
        assert result.metadata["retrieved_chunk_ids"] == []
        assert result.metadata["follow_up_queries"] == []
        assert mock_retrieval._retrieve_by_id.await_count == 0
        assert mock_retrieval.retrieve.await_count == 0

    @pytest.mark.asyncio
    async def test_single_route_uses_single_retrieval(self, session_factory, mock_retrieval, cleanup):
        llm = FakeListLLM(responses=["moderate", "Single answer"])
        pipeline = AdaptiveRAGPipeline(
            session_factory=session_factory,
            name="test_adaptive_rag_single",
            llm=llm,
            retrieval_pipeline=mock_retrieval,
        )
        cleanup.append(pipeline.pipeline_id)

        result = await pipeline._generate(query_id=1, top_k=3)

        assert result.text == "Single answer"
        assert result.metadata is not None
        assert result.metadata["complexity_tier"] == "moderate"
        assert result.metadata["route"] == "single"
        assert result.metadata["retrieved_chunk_ids"] == [1, 2, 3]
        assert result.metadata["retrieved_scores"] == [0.95, 0.85, 0.75]
        assert result.metadata["follow_up_queries"] == []
        mock_retrieval._retrieve_by_id.assert_awaited_once_with(1, 3)
        assert mock_retrieval.retrieve.await_count == 0

    @pytest.mark.asyncio
    async def test_multi_route_performs_ircot_interleaved_retrieval(self, session_factory, mock_retrieval, cleanup):
        llm = FakeListLLM(responses=["complex", "Thought about X.", "So the answer is: 42", "Complex answer"])
        mock_retrieval._retrieve_by_id = AsyncMock(
            return_value=[{"doc_id": 1, "score": 0.8}, {"doc_id": 2, "score": 0.7}]
        )
        mock_retrieval.retrieve = AsyncMock(return_value=[{"doc_id": 4, "score": 0.95}])

        pipeline = AdaptiveRAGPipeline(
            session_factory=session_factory,
            name="test_adaptive_rag_multi",
            llm=llm,
            retrieval_pipeline=mock_retrieval,
            max_multi_steps=2,
        )
        cleanup.append(pipeline.pipeline_id)

        result = await pipeline._generate(query_id=1, top_k=4)

        assert result.text == "Complex answer"
        assert result.metadata is not None
        assert result.metadata["complexity_tier"] == "complex"
        assert result.metadata["route"] == "multi"
        assert result.metadata["cot_sentences"] == ["Thought about X.", "So the answer is: 42"]
        assert result.metadata["steps_completed"] == 2
        assert result.metadata["retrieved_chunk_ids"] == [1, 2, 4]
        assert result.metadata["retrieved_scores"] == [0.8, 0.7, 0.95]
        mock_retrieval._retrieve_by_id.assert_awaited_once_with(1, 4)
        mock_retrieval.retrieve.assert_awaited_once_with("Thought about X.", 4)

    @pytest.mark.asyncio
    async def test_unknown_complexity_falls_back_to_single_route(self, session_factory, mock_retrieval, cleanup):
        llm = FakeListLLM(responses=["uncertain", "Fallback single answer"])
        pipeline = AdaptiveRAGPipeline(
            session_factory=session_factory,
            name="test_adaptive_rag_fallback",
            llm=llm,
            retrieval_pipeline=mock_retrieval,
        )
        cleanup.append(pipeline.pipeline_id)

        result = await pipeline._generate(query_id=1, top_k=2)

        assert result.text == "Fallback single answer"
        assert result.metadata is not None
        assert result.metadata["complexity_tier"] == "moderate"
        assert result.metadata["route"] == "single"
        mock_retrieval._retrieve_by_id.assert_awaited_once_with(1, 2)
        assert mock_retrieval.retrieve.await_count == 0

    @pytest.mark.asyncio
    async def test_multi_route_uses_answer_signal_and_final_prompt_cot_history(self, mock_retrieval):
        class StubService:
            def get_query_text(self, query_id: int | str) -> str:
                return f"query {query_id}"

            def get_chunk_contents(self, chunk_ids: list[int | str]) -> list[str]:
                return [f"chunk {chunk_id}" for chunk_id in chunk_ids]

        llm = FakeListLLM(responses=["complex", "Done: 42", "Complex answer"])
        mock_retrieval._retrieve_by_id = AsyncMock(return_value=[{"doc_id": 1, "score": 0.8}])
        mock_retrieval.retrieve = AsyncMock(return_value=[{"doc_id": 4, "score": 0.95}])
        pipeline = AdaptiveRAGPipeline.__new__(AdaptiveRAGPipeline)
        pipeline._service = StubService()
        pipeline._llm = llm
        pipeline._retrieval_pipeline = mock_retrieval
        pipeline._complexity_prompt_template = "{query}"
        pipeline._zero_retrieval_prompt_template = "{query}"
        pipeline._single_retrieval_prompt_template = "{context} {query}"
        pipeline._multi_reasoning_prompt_template = (
            "Question: {query}\nParagraphs: {paragraphs}\nThoughts: {cot_history}"
        )
        pipeline._multi_retrieval_answer_prompt_template = "{query} {context} {cot_history}"
        pipeline._route_for_simple = "zero"
        pipeline._route_for_moderate = "single"
        pipeline._route_for_complex = "multi"
        pipeline._max_multi_steps = 2
        pipeline._answer_signal = "done:"
        pipeline._paragraph_budget = 15

        result = await pipeline._generate(query_id=1, top_k=4)

        assert result.text == "Complex answer"
        assert result.metadata is not None
        assert result.metadata["cot_sentences"] == ["Done: 42"]
        assert result.metadata["steps_completed"] == 1
        mock_retrieval._retrieve_by_id.assert_awaited_once_with(1, 4)
        assert mock_retrieval.retrieve.await_count == 0

    @pytest.mark.asyncio
    async def test_multi_route_dedups_and_applies_fifo_paragraph_budget(self, mock_retrieval):
        class StubService:
            def get_query_text(self, query_id: int | str) -> str:
                return f"query {query_id}"

            def get_chunk_contents(self, chunk_ids: list[int | str]) -> list[str]:
                return [f"chunk {chunk_id}" for chunk_id in chunk_ids]

        llm = FakeListLLM(responses=["complex", "Thought one.", "Thought two.", "Final answer"])
        mock_retrieval._retrieve_by_id = AsyncMock(
            return_value=[{"doc_id": 1, "score": 0.1}, {"doc_id": 2, "score": 0.2}]
        )
        mock_retrieval.retrieve = AsyncMock(
            side_effect=[
                [{"doc_id": 2, "score": 0.9}, {"doc_id": 3, "score": 0.3}, {"doc_id": 4, "score": 0.4}],
                [{"doc_id": 4, "score": 0.8}, {"doc_id": 5, "score": 0.5}],
            ]
        )
        pipeline = AdaptiveRAGPipeline.__new__(AdaptiveRAGPipeline)
        pipeline._service = StubService()
        pipeline._llm = llm
        pipeline._retrieval_pipeline = mock_retrieval
        pipeline._complexity_prompt_template = "{query}"
        pipeline._zero_retrieval_prompt_template = "{query}"
        pipeline._single_retrieval_prompt_template = "{context} {query}"
        pipeline._multi_reasoning_prompt_template = "{query} {paragraphs} {cot_history}"
        pipeline._multi_retrieval_answer_prompt_template = "{query} {context} {cot_history}"
        pipeline._route_for_simple = "zero"
        pipeline._route_for_moderate = "single"
        pipeline._route_for_complex = "multi"
        pipeline._max_multi_steps = 2
        pipeline._answer_signal = "answer is:"
        pipeline._paragraph_budget = 3

        result = await pipeline._generate(query_id=1, top_k=3)

        assert result.metadata is not None
        assert result.metadata["retrieved_chunk_ids"] == [3, 4, 5]
        assert result.metadata["retrieved_scores"] == [0.3, 0.4, 0.5]
        assert result.metadata["cot_sentences"] == ["Thought one.", "Thought two."]
        assert mock_retrieval.retrieve.await_args_list[0].args == ("Thought one.", 3)
        assert mock_retrieval.retrieve.await_args_list[1].args == ("Thought two.", 3)

    @pytest.mark.asyncio
    async def test_multi_route_dedups_duplicate_chunk_ids_within_one_retrieval(self, mock_retrieval):
        class StubService:
            def get_query_text(self, query_id: int | str) -> str:
                return f"query {query_id}"

            def get_chunk_contents(self, chunk_ids: list[int | str]) -> list[str]:
                return [f"chunk {chunk_id}" for chunk_id in chunk_ids]

        llm = FakeListLLM(responses=["complex", "Thought one.", "Final answer"])
        duplicate_results = [{"doc_id": 1, "score": 0.9}, {"doc_id": 1, "score": 0.8}]
        mock_retrieval._retrieve_by_id = AsyncMock(return_value=duplicate_results)
        mock_retrieval.retrieve = AsyncMock(return_value=duplicate_results)
        pipeline = AdaptiveRAGPipeline.__new__(AdaptiveRAGPipeline)
        pipeline._service = StubService()
        pipeline._llm = llm
        pipeline._retrieval_pipeline = mock_retrieval
        pipeline._complexity_prompt_template = "{query}"
        pipeline._zero_retrieval_prompt_template = "{query}"
        pipeline._single_retrieval_prompt_template = "{context} {query}"
        pipeline._multi_reasoning_prompt_template = "{query} {paragraphs} {cot_history}"
        pipeline._multi_retrieval_answer_prompt_template = "{query} {context} {cot_history}"
        pipeline._route_for_simple = "zero"
        pipeline._route_for_moderate = "single"
        pipeline._route_for_complex = "multi"
        pipeline._max_multi_steps = 1
        pipeline._answer_signal = "answer is:"
        pipeline._paragraph_budget = 5

        result = await pipeline._generate(query_id=1, top_k=3)

        assert result.metadata is not None
        assert result.metadata["retrieved_chunk_ids"] == [1]
        assert result.metadata["retrieved_scores"] == [0.9]

    def test_complexity_parser_uses_safest_tier_when_multiple_labels_are_present(self):
        assert AdaptiveRAGPipeline._parse_complexity_tier("not simple; this is complex") == "complex"

    def test_complexity_parser_accepts_single_label_explanatory_output(self):
        assert AdaptiveRAGPipeline._parse_complexity_tier("The question is moderate.") == "moderate"

    @pytest.mark.parametrize("response_text", ["simplicity", "simplex", "complexity"])
    def test_complexity_parser_matches_labels_as_whole_tokens(self, response_text):
        assert AdaptiveRAGPipeline._parse_complexity_tier(response_text) == "moderate"

    def test_config_rejects_invalid_route_values(self, mock_retrieval):
        llm = FakeListLLM(responses=["answer"])
        invalid_route = cast(Any, "singel")
        config = AdaptiveRAGPipelineConfig(
            name="adaptive_rag_invalid_route_cfg",
            retrieval_pipeline_name="bm25",
            llm=llm,
            route_for_simple=invalid_route,
        )
        config.inject_retrieval_pipeline(mock_retrieval)

        with pytest.raises(ValueError, match="route_for_simple"):
            config.get_pipeline_kwargs()

    def test_adaptive_rag_config(self, mock_retrieval):
        llm = FakeListLLM(responses=["answer"])
        config = AdaptiveRAGPipelineConfig(
            name="adaptive_rag_cfg",
            retrieval_pipeline_name="bm25",
            llm=llm,
            route_for_simple="single",
            route_for_moderate="multi",
            route_for_complex="zero",
            max_multi_steps=4,
            answer_signal="DONE",
            paragraph_budget=7,
        )
        config.inject_retrieval_pipeline(mock_retrieval)

        kwargs = config.get_pipeline_kwargs()

        assert config.get_pipeline_class() is AdaptiveRAGPipeline
        assert kwargs["llm"] is llm
        assert kwargs["retrieval_pipeline"] is mock_retrieval
        assert kwargs["route_for_simple"] == "single"
        assert kwargs["route_for_moderate"] == "multi"
        assert kwargs["route_for_complex"] == "zero"
        assert kwargs["max_multi_steps"] == 4
        assert kwargs["answer_signal"] == "DONE"
        assert kwargs["paragraph_budget"] == 7

    def test_pipeline_config_output(self, session_factory, mock_retrieval, cleanup):
        llm = FakeListLLM(responses=["simple", "answer"])
        pipeline = AdaptiveRAGPipeline(
            session_factory=session_factory,
            name="test_adaptive_rag_config_output",
            llm=llm,
            retrieval_pipeline=mock_retrieval,
            route_for_simple="zero",
            route_for_moderate="single",
            route_for_complex="multi",
            max_multi_steps=3,
            answer_signal="HALT",
            paragraph_budget=9,
        )
        cleanup.append(pipeline.pipeline_id)

        config = pipeline._get_pipeline_config()

        assert config["type"] == "adaptive_rag"
        assert config["retrieval_pipeline_id"] == mock_retrieval.pipeline_id
        assert config["route_for_simple"] == "zero"
        assert config["route_for_moderate"] == "single"
        assert config["route_for_complex"] == "multi"
        assert config["max_multi_steps"] == 3
        assert config["answer_signal"] == "HALT"
        assert config["paragraph_budget"] == 9
        assert "complexity_prompt_template" in config
        assert "zero_retrieval_prompt_template" in config
        assert "single_retrieval_prompt_template" in config
        assert "multi_reasoning_prompt_template" in config
        assert "multi_retrieval_answer_prompt_template" in config

    def test_run_pipeline(self, session_factory, cleanup):
        from autorag_research.orm.repository.query import QueryRepository

        with session_factory() as session:
            query_repo = QueryRepository(session)
            query_count = query_repo.count()

        retrieval = create_mock_retrieval_pipeline()
        llm = create_mock_llm()

        pipeline = AdaptiveRAGPipeline(
            session_factory=session_factory,
            name="test_adaptive_rag_run",
            llm=llm,
            retrieval_pipeline=retrieval,
        )
        cleanup.append(pipeline.pipeline_id)

        result = pipeline.run(top_k=2, batch_size=10)

        config = PipelineTestConfig(
            pipeline_type="generation",
            expected_total_queries=query_count,
            check_token_usage=True,
            check_execution_time=True,
            check_persistence=True,
        )
        assert isinstance(pipeline.pipeline_id, int)
        verifier = PipelineTestVerifier(result, pipeline.pipeline_id, session_factory, config)
        verifier.verify_all()
