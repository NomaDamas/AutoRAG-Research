"""Tests for the Question Decomposition retrieval pipeline."""

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from langchain_core.language_models.fake import FakeListLLM

from autorag_research.config import BaseRetrievalPipelineConfig
from autorag_research.orm.repository.query import QueryRepository
from autorag_research.pipelines.retrieval.question_decomposition import (
    DEFAULT_DECOMPOSITION_PROMPT,
    QuestionDecompositionRetrievalPipeline,
    QuestionDecompositionRetrievalPipelineConfig,
)
from autorag_research.rerankers.base import BaseReranker, RerankResult
from tests.autorag_research.pipelines.pipeline_test_utils import (
    PipelineTestConfig,
    PipelineTestVerifier,
    cleanup_pipeline_results_factory,
)


class FakeReranker(BaseReranker):
    """Simple reranker for testing."""

    model_name: str = "fake-reranker"

    def rerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        results = [RerankResult(index=i, text=doc, score=1.0 - i * 0.1) for i, doc in enumerate(documents)]
        if top_k is not None:
            results = results[:top_k]
        return results

    async def arerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        return self.rerank(query, documents, top_k)


@pytest.fixture
def cleanup(session_factory):
    yield from cleanup_pipeline_results_factory(session_factory)


@pytest.fixture
def mock_inner_pipeline():
    mock = MagicMock()
    mock.pipeline_id = 101
    mock._retrieve_by_id = AsyncMock(return_value=[])
    mock._retrieve_by_text = AsyncMock(return_value=[])
    mock.retrieve = AsyncMock(return_value=[])
    return mock


@pytest.fixture
def question_decomposition_pipeline(session_factory, cleanup, mock_inner_pipeline):
    llm = FakeListLLM(responses=["1. First sub-question?\n2. Second sub-question?"])
    pipeline = QuestionDecompositionRetrievalPipeline(
        session_factory=session_factory,
        name="test_question_decomposition_fixture",
        llm=llm,
        inner_retrieval_pipeline=mock_inner_pipeline,
    )
    cleanup.append(pipeline.pipeline_id)
    return pipeline


class TestParseSubQuestions:
    """Unit tests for sub-question parsing."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("1. What is A?\n2. What is B?", ["What is A?", "What is B?"]),
            ("- Find the author\n- Find the date", ["Find the author", "Find the date"]),
            ("What is A? What is B?", ["What is A?", "What is B?"]),
            ("Single sub-question", ["Single sub-question"]),
            ("", []),
        ],
    )
    def test_parse_subquestions(self, text, expected):
        result = QuestionDecompositionRetrievalPipeline._parse_subquestions(MagicMock(), text)
        assert result == expected


class TestMergeResults:
    """Unit tests for merge behavior."""

    def test_merge_results_keeps_highest_score(self, question_decomposition_pipeline):
        merged = question_decomposition_pipeline._merge_results([
            [{"doc_id": 1, "score": 0.4, "content": "alpha"}, {"doc_id": 2, "score": 0.7, "content": "beta"}],
            [{"doc_id": 1, "score": 0.9, "content": "alpha-new"}, {"doc_id": 3, "score": 0.6, "content": "gamma"}],
        ])

        assert merged == [
            {"doc_id": 1, "score": 0.9, "content": "alpha-new"},
            {"doc_id": 2, "score": 0.7, "content": "beta"},
            {"doc_id": 3, "score": 0.6, "content": "gamma"},
        ]


class TestQuestionDecompositionRetrievalPipelineConfig:
    """Tests for config behavior."""

    def test_config_get_pipeline_class(self):
        llm = FakeListLLM(responses=["sub-question"])
        config = QuestionDecompositionRetrievalPipelineConfig(
            name="test_question_decomposition",
            llm=llm,
            inner_retrieval_pipeline_name="bm25",
        )

        assert config.get_pipeline_class() is QuestionDecompositionRetrievalPipeline

    def test_config_get_pipeline_kwargs(self, mock_inner_pipeline):
        llm = FakeListLLM(responses=["sub-question"])
        reranker = FakeReranker()
        config = QuestionDecompositionRetrievalPipelineConfig(
            name="test_question_decomposition",
            llm=llm,
            inner_retrieval_pipeline_name="bm25",
            reranker=reranker,
            max_subquestions=4,
            fetch_k_multiplier=3,
        )
        config.inject_retrieval_pipeline(mock_inner_pipeline)

        kwargs = config.get_pipeline_kwargs()

        assert kwargs["llm"] is llm
        assert kwargs["inner_retrieval_pipeline"] is mock_inner_pipeline
        assert kwargs["reranker"] is reranker
        assert kwargs["max_subquestions"] == 4
        assert kwargs["fetch_k_multiplier"] == 3

    def test_config_raises_without_injected_pipeline(self):
        llm = FakeListLLM(responses=["sub-question"])
        config = QuestionDecompositionRetrievalPipelineConfig(
            name="test_question_decomposition",
            llm=llm,
            inner_retrieval_pipeline_name="bm25",
        )

        with pytest.raises(ValueError, match="not injected"):
            config.get_pipeline_kwargs()

    @pytest.mark.api
    def test_config_string_model_conversion(self, mock_inner_pipeline):
        with (
            patch("autorag_research.injection.load_llm") as mock_load_llm,
            patch("autorag_research.injection.load_reranker") as mock_load_reranker,
        ):
            llm = MagicMock()
            reranker = FakeReranker()
            mock_load_llm.return_value = llm
            mock_load_reranker.return_value = reranker

            config = QuestionDecompositionRetrievalPipelineConfig(
                name="test_question_decomposition",
                llm="mock",
                inner_retrieval_pipeline_name="bm25",
                reranker="sentence_transformer",
            )
            config.inject_retrieval_pipeline(mock_inner_pipeline)

            assert config.llm is llm
            assert config.reranker is reranker
            mock_load_llm.assert_called_once_with("mock")
            mock_load_reranker.assert_called_once_with("sentence_transformer")


class TestQuestionDecompositionRetrievalPipeline:
    """Tests for retrieval pipeline behavior."""

    def test_pipeline_creation(self, session_factory, cleanup, mock_inner_pipeline):
        llm = FakeListLLM(responses=["sub-question"])

        pipeline = QuestionDecompositionRetrievalPipeline(
            session_factory=session_factory,
            name="test_question_decomposition_creation",
            llm=llm,
            inner_retrieval_pipeline=mock_inner_pipeline,
            max_subquestions=4,
            fetch_k_multiplier=3,
        )
        cleanup.append(pipeline.pipeline_id)

        assert pipeline.pipeline_id > 0
        assert pipeline.max_subquestions == 4
        assert pipeline.fetch_k_multiplier == 3

    def test_pipeline_config(self, session_factory, cleanup, mock_inner_pipeline):
        llm = FakeListLLM(responses=["sub-question"])
        reranker = FakeReranker()

        pipeline = QuestionDecompositionRetrievalPipeline(
            session_factory=session_factory,
            name="test_question_decomposition_config",
            llm=llm,
            inner_retrieval_pipeline=mock_inner_pipeline,
            reranker=reranker,
        )
        cleanup.append(pipeline.pipeline_id)

        config = pipeline._get_pipeline_config()
        assert config == {
            "type": "question_decomposition",
            "max_subquestions": 3,
            "fetch_k_multiplier": 2,
            "decomposition_prompt_template": DEFAULT_DECOMPOSITION_PROMPT,
            "inner_retrieval_pipeline_id": 101,
            "reranker_model": "fake-reranker",
        }

    @pytest.mark.asyncio
    async def test_retrieve_by_id_decomposes_and_retrieves(self, session_factory, cleanup, mock_inner_pipeline):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(content="1. Who wrote it?\n2. When was it published?"))

        mock_inner_pipeline._retrieve_by_id.return_value = [{"doc_id": 1, "score": 0.7, "content": "doc one"}]
        mock_inner_pipeline.retrieve.side_effect = [
            [{"doc_id": 2, "score": 0.95, "content": "doc two"}],
            [{"doc_id": 3, "score": 0.8, "content": "doc three"}],
        ]

        pipeline = QuestionDecompositionRetrievalPipeline(
            session_factory=session_factory,
            name="test_question_decomposition_retrieve_by_id",
            llm=llm,
            inner_retrieval_pipeline=mock_inner_pipeline,
        )
        cleanup.append(pipeline.pipeline_id)

        results = await pipeline._retrieve_by_id(1, top_k=2)

        llm.ainvoke.assert_awaited_once()
        mock_inner_pipeline._retrieve_by_id.assert_awaited_once_with(1, 4)
        assert mock_inner_pipeline.retrieve.await_count == 2
        assert mock_inner_pipeline.retrieve.await_args_list[0].args == ("Who wrote it?", 4)
        assert mock_inner_pipeline.retrieve.await_args_list[1].args == ("When was it published?", 4)
        assert results == [
            {"doc_id": 2, "score": 0.95, "content": "doc two"},
            {"doc_id": 3, "score": 0.8, "content": "doc three"},
        ]

    @pytest.mark.asyncio
    async def test_retrieve_by_id_with_reranker(self, session_factory, cleanup, mock_inner_pipeline):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(content="1. Who wrote it?"))
        reranker = FakeReranker()

        mock_inner_pipeline._retrieve_by_id.return_value = [
            {"doc_id": 11, "score": 0.6, "content": "first doc"},
            {"doc_id": 12, "score": 0.5, "content": "second doc"},
        ]
        mock_inner_pipeline.retrieve.return_value = [{"doc_id": 13, "score": 0.9, "content": "third doc"}]

        pipeline = QuestionDecompositionRetrievalPipeline(
            session_factory=session_factory,
            name="test_question_decomposition_rerank",
            llm=llm,
            inner_retrieval_pipeline=mock_inner_pipeline,
            reranker=reranker,
        )
        cleanup.append(pipeline.pipeline_id)

        results = await pipeline._retrieve_by_id(1, top_k=2)

        assert results == [
            {"doc_id": 13, "score": 1.0, "content": "third doc"},
            {"doc_id": 11, "score": 0.9, "content": "first doc"},
        ]

    @pytest.mark.asyncio
    async def test_retrieve_by_id_without_reranker(self, session_factory, cleanup, mock_inner_pipeline):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(content="1. Who wrote it?"))

        mock_inner_pipeline._retrieve_by_id.return_value = [{"doc_id": 1, "score": 0.2, "content": "doc one"}]
        mock_inner_pipeline.retrieve.return_value = [{"doc_id": 2, "score": 0.9, "content": "doc two"}]

        pipeline = QuestionDecompositionRetrievalPipeline(
            session_factory=session_factory,
            name="test_question_decomposition_no_rerank",
            llm=llm,
            inner_retrieval_pipeline=mock_inner_pipeline,
            reranker=None,
        )
        cleanup.append(pipeline.pipeline_id)

        results = await pipeline._retrieve_by_id(1, top_k=1)

        assert results == [{"doc_id": 2, "score": 0.9, "content": "doc two"}]

    @pytest.mark.asyncio
    async def test_deduplicate_docs_keeps_highest_score(self, session_factory, cleanup, mock_inner_pipeline):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(content="1. Who wrote it?"))

        mock_inner_pipeline._retrieve_by_id.return_value = [{"doc_id": 1, "score": 0.3, "content": "doc one"}]
        mock_inner_pipeline.retrieve.return_value = [
            {"doc_id": 1, "score": 0.8, "content": "doc one better"},
            {"doc_id": 2, "score": 0.7, "content": "doc two"},
        ]

        pipeline = QuestionDecompositionRetrievalPipeline(
            session_factory=session_factory,
            name="test_question_decomposition_dedup",
            llm=llm,
            inner_retrieval_pipeline=mock_inner_pipeline,
        )
        cleanup.append(pipeline.pipeline_id)

        results = await pipeline._retrieve_by_id(1, top_k=2)

        assert results == [
            {"doc_id": 1, "score": 0.8, "content": "doc one better"},
            {"doc_id": 2, "score": 0.7, "content": "doc two"},
        ]

    @pytest.mark.asyncio
    async def test_single_query_retrieve(self, session_factory, cleanup, mock_inner_pipeline):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(content="1. Clarify this?"))

        mock_inner_pipeline._retrieve_by_text.side_effect = [
            [{"doc_id": 21, "score": 0.9, "content": "base result"}],
            [{"doc_id": 22, "score": 0.8, "content": "sub-question result"}],
        ]

        pipeline = QuestionDecompositionRetrievalPipeline(
            session_factory=session_factory,
            name="test_question_decomposition_single_query",
            llm=llm,
            inner_retrieval_pipeline=mock_inner_pipeline,
        )
        cleanup.append(pipeline.pipeline_id)

        results = await pipeline.retrieve("ad hoc query not in seed data", top_k=2)

        assert mock_inner_pipeline._retrieve_by_text.await_count == 2
        assert results == [
            {"doc_id": 21, "score": 0.9, "content": "base result"},
            {"doc_id": 22, "score": 0.8, "content": "sub-question result"},
        ]

    def test_run_full_pipeline(self, session_factory, cleanup, mock_inner_pipeline):
        session = session_factory()
        try:
            query_count = QueryRepository(session).count()
        finally:
            session.close()

        llm = FakeListLLM(responses=["1. Find supporting detail?\n2. Find date?"] * query_count)
        mock_inner_pipeline._retrieve_by_id.return_value = [
            {"doc_id": 1, "score": 0.9, "content": "Chunk 1-1"},
            {"doc_id": 2, "score": 0.8, "content": "Chunk 1-2"},
        ]
        mock_inner_pipeline.retrieve.return_value = [
            {"doc_id": 3, "score": 0.7, "content": "Chunk 2-1"},
            {"doc_id": 4, "score": 0.6, "content": "Chunk 2-2"},
        ]

        pipeline = QuestionDecompositionRetrievalPipeline(
            session_factory=session_factory,
            name=f"test_question_decomposition_run_full_{uuid4().hex}",
            llm=llm,
            inner_retrieval_pipeline=mock_inner_pipeline,
        )
        cleanup.append(pipeline.pipeline_id)

        result = pipeline.run(top_k=3)

        config = PipelineTestConfig(
            pipeline_type="retrieval",
            expected_total_queries=query_count,
            expected_min_results=0,
            check_persistence=True,
        )
        verifier = PipelineTestVerifier(result, pipeline.pipeline_id, session_factory, config)
        verifier.verify_all()


class TestExecutorDependencyResolution:
    """Focused executor test for nested retrieval dependencies."""

    def test_executor_injects_inner_retrieval_pipeline(self, session_factory):
        from autorag_research.config import ExecutorConfig
        from autorag_research.executor import Executor

        @dataclass(kw_only=True)
        class MockNestedRetrievalConfig(BaseRetrievalPipelineConfig):
            inner_retrieval_pipeline_name: str
            _inner_retrieval_pipeline: Any | None = field(default=None, repr=False)

            def get_pipeline_class(self) -> type:
                return MagicMock

            def get_pipeline_kwargs(self) -> dict[str, Any]:
                return {}

            def inject_retrieval_pipeline(self, pipeline: Any) -> None:
                self._inner_retrieval_pipeline = pipeline

        config = MockNestedRetrievalConfig(
            name="test_nested_retrieval",
            inner_retrieval_pipeline_name="bm25",
        )
        executor = Executor(session_factory, ExecutorConfig(pipelines=[], metrics=[]))
        executor._schema = None

        with (
            patch.object(executor._config_resolver, "resolve_config") as mock_resolve,
            patch("autorag_research.pipelines.retrieval.loader.instantiate") as mock_instantiate,
        ):
            mock_resolve.return_value = {"_target_": "unused"}
            mock_pipeline_config = MagicMock()
            mock_pipeline_config.name = "bm25"
            mock_pipeline_config.get_pipeline_class.return_value = MagicMock(return_value="pipeline-instance")
            mock_pipeline_config.get_pipeline_kwargs.return_value = {}
            mock_instantiate.return_value = mock_pipeline_config

            executor._resolve_dependencies(config)

        assert config._inner_retrieval_pipeline == "pipeline-instance"
        assert executor._dependency_pipelines["bm25"] == "pipeline-instance"
