"""Tests for the Query Rewrite retrieval pipeline."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.language_models.fake import FakeListLLM
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.orm.repository.chunk_retrieved_result import ChunkRetrievedResultRepository
from autorag_research.pipelines.retrieval.query_rewrite import (
    DEFAULT_QUERY_REWRITE_PROMPT_TEMPLATE,
    QueryRewritePipelineConfig,
    QueryRewriteRetrievalPipeline,
)
from tests.autorag_research.pipelines.pipeline_test_utils import (
    PipelineTestConfig,
    PipelineTestVerifier,
    create_mock_retrieval_pipeline,
)


@pytest.fixture
def cleanup_pipeline_results(session_factory: sessionmaker[Session]):
    """Cleanup fixture that deletes retrieval pipeline results after test."""
    created_pipeline_ids: list[int] = []

    yield created_pipeline_ids

    session = session_factory()
    try:
        result_repo = ChunkRetrievedResultRepository(session)
        for pipeline_id in created_pipeline_ids:
            result_repo.delete_by_pipeline(pipeline_id)
        session.commit()
    finally:
        session.close()


class TestQueryRewritePipelineConfig:
    """Tests for QueryRewritePipelineConfig."""

    def test_get_pipeline_class(self):
        llm = FakeListLLM(responses=["rewritten query"])
        config = QueryRewritePipelineConfig(
            name="query_rewrite",
            llm=llm,
            retrieval_pipeline_name="bm25",
        )

        assert config.get_pipeline_class() == QueryRewriteRetrievalPipeline

    def test_default_prompt_template(self):
        llm = FakeListLLM(responses=["rewritten query"])
        config = QueryRewritePipelineConfig(
            name="query_rewrite",
            llm=llm,
            retrieval_pipeline_name="bm25",
        )

        assert config.prompt_template == DEFAULT_QUERY_REWRITE_PROMPT_TEMPLATE

    def test_get_pipeline_kwargs_requires_injected_retrieval_pipeline(self):
        llm = FakeListLLM(responses=["rewritten query"])
        config = QueryRewritePipelineConfig(
            name="query_rewrite",
            llm=llm,
            retrieval_pipeline_name="bm25",
        )

        with pytest.raises(ValueError, match="not injected"):
            config.get_pipeline_kwargs()

    def test_get_pipeline_kwargs_after_injection(self):
        llm = FakeListLLM(responses=["rewritten query"])
        wrapped_retrieval = create_mock_retrieval_pipeline(pipeline_id=77)
        config = QueryRewritePipelineConfig(
            name="query_rewrite",
            llm=llm,
            retrieval_pipeline_name="bm25",
            prompt_template="Rewrite: {query}",
        )

        config.inject_retrieval_pipeline(wrapped_retrieval)
        kwargs = config.get_pipeline_kwargs()

        assert kwargs["llm"] is llm
        assert kwargs["retrieval_pipeline"] is wrapped_retrieval
        assert kwargs["prompt_template"] == "Rewrite: {query}"

    @pytest.mark.api
    def test_string_llm_conversion(self):
        with patch("autorag_research.injection.load_llm") as mock_load:
            mock_llm = MagicMock()
            mock_load.return_value = mock_llm

            config = QueryRewritePipelineConfig(
                name="query_rewrite",
                llm="mock",
                retrieval_pipeline_name="bm25",
            )

            mock_load.assert_called_once_with("mock")
            assert config.llm is mock_llm


class TestQueryRewriteRetrievalPipeline:
    """Tests for QueryRewriteRetrievalPipeline."""

    def test_creation_rejects_missing_query_placeholder(self, session_factory):
        with pytest.raises(ValueError, match="\\{query\\}"):
            QueryRewriteRetrievalPipeline(
                session_factory=session_factory,
                name="query_rewrite_invalid_prompt",
                llm=FakeListLLM(responses=["rewritten query"]),
                retrieval_pipeline=create_mock_retrieval_pipeline(),
                prompt_template="Rewrite without placeholder",
            )

    @pytest.mark.asyncio
    async def test_rewrite_query_uses_prompt_template(self, session_factory, cleanup_pipeline_results: list[int]):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value="rewritten query")

        pipeline = QueryRewriteRetrievalPipeline(
            session_factory=session_factory,
            name="query_rewrite_template_usage",
            llm=llm,
            retrieval_pipeline=create_mock_retrieval_pipeline(),
            prompt_template="Rewrite better: {query}",
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        rewritten = await pipeline._rewrite_query("original query")

        assert rewritten == "rewritten query"
        llm.ainvoke.assert_awaited_once_with("Rewrite better: original query")

    @pytest.mark.asyncio
    async def test_retrieve_by_text_rewrites_then_delegates(self, session_factory, cleanup_pipeline_results: list[int]):
        wrapped_retrieval = create_mock_retrieval_pipeline(default_results=[{"doc_id": 9, "score": 0.95}])
        pipeline = QueryRewriteRetrievalPipeline(
            session_factory=session_factory,
            name="query_rewrite_by_text",
            llm=FakeListLLM(responses=["rewritten query"]),
            retrieval_pipeline=wrapped_retrieval,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        results = await pipeline._retrieve_by_text("original query", top_k=4)

        wrapped_retrieval.retrieve.assert_awaited_once_with("rewritten query", 4)
        assert results == [{"doc_id": 9, "score": 0.95}]

    @pytest.mark.asyncio
    async def test_retrieve_by_id_fetches_query_text_then_rewrites(
        self,
        session_factory,
        cleanup_pipeline_results: list[int],
    ):
        wrapped_retrieval = create_mock_retrieval_pipeline(default_results=[{"doc_id": 3, "score": 0.88}])
        pipeline = QueryRewriteRetrievalPipeline(
            session_factory=session_factory,
            name="query_rewrite_by_id",
            llm=FakeListLLM(responses=["rewritten seeded query"]),
            retrieval_pipeline=wrapped_retrieval,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        results = await pipeline._retrieve_by_id(1, top_k=2)

        wrapped_retrieval.retrieve.assert_awaited_once_with("rewritten seeded query", 2)
        assert results == [{"doc_id": 3, "score": 0.88}]

    def test_run_full_pipeline(self, session_factory, cleanup_pipeline_results: list[int]):
        from autorag_research.orm.repository.query import QueryRepository

        session = session_factory()
        try:
            query_count = QueryRepository(session).count()
        finally:
            session.close()

        wrapped_retrieval = create_mock_retrieval_pipeline(
            default_results=[
                {"doc_id": 1, "score": 0.91, "content": "Content 1"},
                {"doc_id": 2, "score": 0.83, "content": "Content 2"},
            ]
        )
        pipeline = QueryRewriteRetrievalPipeline(
            session_factory=session_factory,
            name="query_rewrite_full_run",
            llm=FakeListLLM(responses=["rewritten query"] * max(query_count, 1)),
            retrieval_pipeline=wrapped_retrieval,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        result = pipeline.run(top_k=2)

        verifier = PipelineTestVerifier(
            result,
            pipeline.pipeline_id,
            session_factory,
            PipelineTestConfig(
                pipeline_type="retrieval",
                expected_total_queries=query_count,
                expected_min_results=0,
                check_persistence=True,
            ),
        )
        verifier.verify_all()
