"""Tests for the generic retriever-reranker retrieval pipeline."""

from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import Field
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.orm.repository.chunk import ChunkRepository
from autorag_research.orm.repository.chunk_retrieved_result import ChunkRetrievedResultRepository
from autorag_research.orm.repository.query import QueryRepository
from autorag_research.pipelines.retrieval.rerank import RerankRetrievalPipeline, RerankRetrievalPipelineConfig
from autorag_research.rerankers.base import BaseReranker, RerankResult
from tests.autorag_research.pipelines.pipeline_test_utils import (
    PipelineTestConfig,
    PipelineTestVerifier,
    create_mock_retrieval_pipeline,
)


class FakeReranker(BaseReranker):
    """Deterministic test reranker that scores documents by exact text."""

    scores_by_text: dict[str, float] = Field(default_factory=dict)

    def rerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        results = [
            RerankResult(index=index, text=document, score=self.scores_by_text.get(document, 0.0))
            for index, document in enumerate(documents)
        ]
        results.sort(key=lambda result: result.score, reverse=True)
        return results[:top_k] if top_k is not None else results

    async def arerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        return self.rerank(query, documents, top_k)


@pytest.fixture
def cleanup_pipeline_results(session_factory: sessionmaker[Session]):
    """Cleanup fixture that deletes rerank pipeline results after test."""
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


class TestRerankRetrievalPipelineConfig:
    """Tests for RerankRetrievalPipelineConfig."""

    def test_get_pipeline_class(self):
        config = RerankRetrievalPipelineConfig(
            name="rerank",
            retrieval_pipeline_name="bm25",
            reranker=FakeReranker(),
        )

        assert config.get_pipeline_class() == RerankRetrievalPipeline

    def test_get_pipeline_kwargs_requires_injected_retrieval_pipeline(self):
        config = RerankRetrievalPipelineConfig(
            name="rerank",
            retrieval_pipeline_name="bm25",
            reranker=FakeReranker(),
        )

        with pytest.raises(ValueError, match="not injected"):
            config.get_pipeline_kwargs()

    def test_get_pipeline_kwargs_after_injection(self):
        wrapped_retrieval = create_mock_retrieval_pipeline(pipeline_id=77)
        reranker = FakeReranker(scores_by_text={"doc": 1.0})
        config = RerankRetrievalPipelineConfig(
            name="rerank",
            retrieval_pipeline_name="bm25",
            reranker=reranker,
            candidate_top_k=25,
        )

        config.inject_retrieval_pipeline(wrapped_retrieval)
        kwargs = config.get_pipeline_kwargs()

        assert kwargs["retrieval_pipeline"] is wrapped_retrieval
        assert kwargs["reranker"] is reranker
        assert kwargs["candidate_top_k"] == 25

    def test_inject_retrieval_pipeline_rejects_image_chunk_pipeline(self):
        wrapped_retrieval = create_mock_retrieval_pipeline(pipeline_id=77)
        wrapped_retrieval._get_pipeline_config.return_value = {"type": "heaven", "retrieval_unit": "image_chunk"}
        config = RerankRetrievalPipelineConfig(
            name="rerank",
            retrieval_pipeline_name="heaven",
            reranker=FakeReranker(),
        )

        with pytest.raises(ValueError, match="only supports text chunk retrieval pipelines"):
            config.inject_retrieval_pipeline(wrapped_retrieval)

    def test_inject_retrieval_pipeline_rejects_missing_retrieval_unit(self):
        wrapped_retrieval = create_mock_retrieval_pipeline(pipeline_id=77)
        wrapped_retrieval._get_pipeline_config.return_value = {
            "type": "query_rewrite",
            "wrapped_pipeline_type": "HEAVENRetrievalPipeline",
        }
        config = RerankRetrievalPipelineConfig(
            name="rerank",
            retrieval_pipeline_name="query_rewrite_heaven",
            reranker=FakeReranker(),
        )

        with pytest.raises(ValueError, match="must declare retrieval_unit='chunk'"):
            config.inject_retrieval_pipeline(wrapped_retrieval)

    @pytest.mark.api
    def test_string_reranker_conversion(self):
        with (
            patch("autorag_research.injection.load_reranker") as mock_load,
            patch(
                "autorag_research.pipelines.retrieval.rerank.health_check_reranker", create=True
            ) as mock_health_check,
        ):
            mock_reranker = MagicMock()
            mock_load.return_value = mock_reranker

            config = RerankRetrievalPipelineConfig(
                name="rerank",
                retrieval_pipeline_name="bm25",
                reranker="mock",
            )

            mock_load.assert_called_once_with("mock")
            mock_health_check.assert_not_called()
            assert config.reranker is mock_reranker


class TestRerankRetrievalPipeline:
    """Tests for RerankRetrievalPipeline."""

    def test_creation_rejects_invalid_candidate_top_k(self, session_factory):
        with (
            patch("autorag_research.pipelines.retrieval.base.BaseRetrievalPipeline.__init__", return_value=None),
            pytest.raises(ValueError, match="candidate_top_k must be >= 1"),
        ):
            RerankRetrievalPipeline(
                session_factory=session_factory,
                name="rerank_invalid_candidate_top_k",
                retrieval_pipeline=create_mock_retrieval_pipeline(),
                reranker=FakeReranker(),
                candidate_top_k=0,
            )

    def test_pipeline_config(self, session_factory, cleanup_pipeline_results: list[int]):
        pipeline = RerankRetrievalPipeline(
            session_factory=session_factory,
            name="rerank_config",
            retrieval_pipeline=create_mock_retrieval_pipeline(pipeline_id=123),
            reranker=FakeReranker(model_name="fake-reranker"),
            candidate_top_k=50,
        )
        cleanup_pipeline_results.append(cast("int", pipeline.pipeline_id))

        config = pipeline._get_pipeline_config()

        assert config["type"] == "rerank"
        assert config["candidate_top_k"] == 50
        assert config["reranker_model"] == "fake-reranker"
        assert config["retrieval_pipeline_id"] == 123
        assert config["retrieval_unit"] == "chunk"

    def test_pipeline_config_declares_text_chunk_retrieval_unit_without_database(self, session_factory):
        with patch("autorag_research.pipelines.retrieval.base.BaseRetrievalPipeline.__init__", return_value=None):
            pipeline = RerankRetrievalPipeline(
                session_factory=session_factory,
                name="rerank_config",
                retrieval_pipeline=create_mock_retrieval_pipeline(pipeline_id=123),
                reranker=FakeReranker(model_name="fake-reranker"),
                candidate_top_k=50,
            )

        config = pipeline._get_pipeline_config()

        assert config["retrieval_unit"] == "chunk"

    def test_creation_rejects_image_chunk_wrapped_pipeline(self, session_factory):
        wrapped_retrieval = create_mock_retrieval_pipeline()
        wrapped_retrieval._get_pipeline_config.return_value = {"type": "heaven", "retrieval_unit": "image_chunk"}

        with (
            patch("autorag_research.pipelines.retrieval.base.BaseRetrievalPipeline.__init__", return_value=None),
            pytest.raises(ValueError, match="only supports text chunk retrieval pipelines"),
        ):
            RerankRetrievalPipeline(
                session_factory=session_factory,
                name="rerank_image_chunk",
                retrieval_pipeline=wrapped_retrieval,
                reranker=FakeReranker(),
            )

    def test_creation_rejects_wrapper_chain_without_explicit_text_retrieval_unit(self, session_factory):
        wrapped_retrieval = create_mock_retrieval_pipeline()
        wrapped_retrieval._get_pipeline_config.return_value = {
            "type": "query_rewrite",
            "wrapped_pipeline_type": "HEAVENRetrievalPipeline",
        }

        with (
            patch("autorag_research.pipelines.retrieval.base.BaseRetrievalPipeline.__init__", return_value=None),
            pytest.raises(ValueError, match="must declare retrieval_unit='chunk'"),
        ):
            RerankRetrievalPipeline(
                session_factory=session_factory,
                name="rerank_missing_retrieval_unit",
                retrieval_pipeline=wrapped_retrieval,
                reranker=FakeReranker(),
            )

    @pytest.mark.asyncio
    async def test_retrieve_by_text_reranks_wrapped_candidates(
        self, session_factory, cleanup_pipeline_results: list[int]
    ):
        wrapped_retrieval = create_mock_retrieval_pipeline(
            default_results=[
                {"doc_id": 1, "score": 0.99, "content": "first document"},
                {"doc_id": 2, "score": 0.40, "content": "second document"},
            ]
        )
        reranker = FakeReranker(scores_by_text={"first document": 0.2, "second document": 0.9})
        pipeline = RerankRetrievalPipeline(
            session_factory=session_factory,
            name="rerank_by_text",
            retrieval_pipeline=wrapped_retrieval,
            reranker=reranker,
            candidate_top_k=2,
        )
        cleanup_pipeline_results.append(cast("int", pipeline.pipeline_id))

        results = await pipeline._retrieve_by_text("original query", top_k=1)

        wrapped_retrieval.retrieve.assert_awaited_once_with("original query", 2)
        assert results == [{"doc_id": 2, "score": 0.9, "content": "second document"}]

    @pytest.mark.asyncio
    async def test_retrieve_by_text_backfills_missing_candidate_content(
        self,
        session_factory,
        cleanup_pipeline_results: list[int],
    ):
        wrapped_retrieval = create_mock_retrieval_pipeline(
            default_results=[
                {"doc_id": 1, "score": 0.9},
                {"doc_id": 2, "score": 0.8},
            ]
        )

        session = session_factory()
        try:
            chunk_repo = ChunkRepository(session)
            chunk_contents = [chunk.contents for chunk in chunk_repo.get_by_ids([1, 2])]
        finally:
            session.close()

        reranker = MagicMock()
        reranker.model_name = "recording-reranker"
        reranker.arerank = AsyncMock(
            return_value=[
                RerankResult(index=1, text=chunk_contents[1], score=0.75),
                RerankResult(index=0, text=chunk_contents[0], score=0.55),
            ]
        )
        pipeline = RerankRetrievalPipeline(
            session_factory=session_factory,
            name="rerank_backfill_content",
            retrieval_pipeline=wrapped_retrieval,
            reranker=reranker,
            candidate_top_k=2,
        )
        cleanup_pipeline_results.append(cast("int", pipeline.pipeline_id))

        await pipeline._retrieve_by_text("query text", top_k=2)

        reranker.arerank.assert_awaited_once_with("query text", chunk_contents, top_k=2)

    def test_missing_candidate_content_requires_existing_chunk(self, session_factory):
        wrapped_retrieval = create_mock_retrieval_pipeline()

        with patch("autorag_research.pipelines.retrieval.base.BaseRetrievalPipeline.__init__", return_value=None):
            pipeline = RerankRetrievalPipeline(
                session_factory=session_factory,
                name="rerank_missing_content",
                retrieval_pipeline=wrapped_retrieval,
                reranker=FakeReranker(),
                candidate_top_k=2,
            )
        pipeline.session_factory = session_factory
        pipeline._schema = None

        mock_uow = MagicMock()
        mock_uow.chunks.get_by_ids.return_value = []
        mock_uow_context = MagicMock()
        mock_uow_context.__enter__.return_value = mock_uow
        mock_uow_context.__exit__.return_value = None

        with (
            patch("autorag_research.pipelines.retrieval.rerank.RetrievalUnitOfWork", return_value=mock_uow_context),
            pytest.raises(ValueError, match="Missing chunk content for candidate doc_ids: 999"),
        ):
            pipeline._ensure_candidate_contents([{"doc_id": 999, "score": 0.9}])

        mock_uow.chunks.get_by_ids.assert_called_once_with([999])

    @pytest.mark.asyncio
    async def test_retrieve_by_id_fetches_query_text_then_reranks(
        self,
        session_factory,
        cleanup_pipeline_results: list[int],
    ):
        wrapped_retrieval = create_mock_retrieval_pipeline(
            default_results=[{"doc_id": 3, "score": 0.88, "content": "doc"}]
        )
        pipeline = RerankRetrievalPipeline(
            session_factory=session_factory,
            name="rerank_by_id",
            retrieval_pipeline=wrapped_retrieval,
            reranker=FakeReranker(scores_by_text={"doc": 0.88}),
        )
        cleanup_pipeline_results.append(cast("int", pipeline.pipeline_id))

        results = await pipeline._retrieve_by_id(1, top_k=2)

        wrapped_retrieval.retrieve.assert_awaited_once()
        assert results == [{"doc_id": 3, "score": 0.88, "content": "doc"}]

    @pytest.mark.asyncio
    async def test_retrieve_by_text_rejects_out_of_range_reranker_index(self, session_factory):
        wrapped_retrieval = create_mock_retrieval_pipeline(
            default_results=[{"doc_id": 1, "score": 0.9, "content": "doc"}]
        )
        reranker = MagicMock()
        reranker.arerank = AsyncMock(return_value=[RerankResult(index=5, text="doc", score=1.0)])

        with patch("autorag_research.pipelines.retrieval.base.BaseRetrievalPipeline.__init__", return_value=None):
            pipeline = RerankRetrievalPipeline(
                session_factory=session_factory,
                name="rerank_bad_index",
                retrieval_pipeline=wrapped_retrieval,
                reranker=reranker,
            )

        with pytest.raises(ValueError, match="out-of-range candidate index"):
            await pipeline._retrieve_by_text("query text", top_k=1)

    def test_run_full_pipeline(self, session_factory, cleanup_pipeline_results: list[int]):
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
        pipeline = RerankRetrievalPipeline(
            session_factory=session_factory,
            name="rerank_full_run",
            retrieval_pipeline=wrapped_retrieval,
            reranker=FakeReranker(scores_by_text={"Content 1": 0.8, "Content 2": 0.6}),
            candidate_top_k=2,
        )
        cleanup_pipeline_results.append(cast("int", pipeline.pipeline_id))

        result = pipeline.run(top_k=2)

        verifier = PipelineTestVerifier(
            result,
            cast("int", pipeline.pipeline_id),
            session_factory,
            PipelineTestConfig(
                pipeline_type="retrieval",
                expected_total_queries=query_count,
                expected_min_results=0,
                check_persistence=True,
            ),
        )
        verifier.verify_all()
