"""Tests for GQR hybrid retrieval pipeline."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.pipelines.retrieval.base import BaseRetrievalPipeline
from autorag_research.pipelines.retrieval.gqr_hybrid import (
    GQRHybridRetrievalPipeline,
    GQRHybridRetrievalPipelineConfig,
)


def _make_stub_pipeline(name: str, by_id_results: list[dict], by_text_results: list[dict]) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        _retrieve_by_id=AsyncMock(return_value=by_id_results),
        _retrieve_by_text=AsyncMock(return_value=by_text_results),
    )


class TestGQRHybridRetrievalPipelineConfig:
    """Tests for GQRHybridRetrievalPipelineConfig."""

    def test_get_pipeline_class(self):
        config = GQRHybridRetrievalPipelineConfig(
            name="gqr_test",
            primary_retrieval_pipeline_name="vector_search",
            complementary_retrieval_pipeline_name="bm25",
        )

        assert config.get_pipeline_class() == GQRHybridRetrievalPipeline

    def test_get_pipeline_kwargs(self):
        config = GQRHybridRetrievalPipelineConfig(
            name="gqr_test",
            primary_retrieval_pipeline_name="vector_search",
            complementary_retrieval_pipeline_name="bm25",
            fetch_k_multiplier=3,
            n_steps=10,
            learning_rate=0.3,
            temperature=0.8,
            mixture_alpha=0.2,
            candidate_pool_mode="union",
        )

        assert config.get_pipeline_kwargs() == {
            "primary_retrieval_pipeline": "vector_search",
            "complementary_retrieval_pipeline": "bm25",
            "fetch_k_multiplier": 3,
            "n_steps": 10,
            "learning_rate": 0.3,
            "temperature": 0.8,
            "mixture_alpha": 0.2,
            "candidate_pool_mode": "union",
        }


class TestGQRHybridRetrievalPipeline:
    """Tests for GQRHybridRetrievalPipeline behavior."""

    @pytest.fixture(autouse=True)
    def _patch_base_pipeline_init(self, monkeypatch: pytest.MonkeyPatch):
        def _fake_base_init(self, session_factory, name, schema=None):
            self.session_factory = session_factory
            self.name = name
            self._schema = schema
            self.pipeline_id = 1
            self._is_new_pipeline = True
            self._service = SimpleNamespace(_create_uow=lambda: None)

        monkeypatch.setattr(BaseRetrievalPipeline, "__init__", _fake_base_init)

    def test_pipeline_creation_with_names(self, session_factory: sessionmaker[Session]):
        with patch("autorag_research.pipelines.retrieval.hybrid.HybridRetrievalPipeline._load_pipeline") as mock_load:
            primary = _make_stub_pipeline(
                "vector_search",
                by_id_results=[{"doc_id": 1, "score": 0.9}],
                by_text_results=[{"doc_id": 1, "score": 0.9}],
            )
            complementary = _make_stub_pipeline(
                "bm25",
                by_id_results=[{"doc_id": 1, "score": 3.0}],
                by_text_results=[{"doc_id": 1, "score": 3.0}],
            )
            mock_load.side_effect = [primary, complementary]

            pipeline = GQRHybridRetrievalPipeline(
                session_factory=session_factory,
                name="gqr_creation",
                primary_retrieval_pipeline="vector_search",
                complementary_retrieval_pipeline="bm25",
            )

            assert mock_load.call_count == 2
            assert pipeline._primary_retrieval_pipeline == primary
            assert pipeline._complementary_retrieval_pipeline == complementary

    @pytest.mark.asyncio
    async def test_candidate_pool_mode_primary_keeps_primary_candidates_only(
        self,
        session_factory: sessionmaker[Session],
    ):
        primary = _make_stub_pipeline(
            "vector_search",
            by_id_results=[{"doc_id": 1, "score": 0.9}, {"doc_id": 2, "score": 0.8}],
            by_text_results=[],
        )
        complementary = _make_stub_pipeline(
            "bm25",
            by_id_results=[{"doc_id": 3, "score": 10.0}],
            by_text_results=[],
        )
        pipeline = GQRHybridRetrievalPipeline(
            session_factory=session_factory,
            name="gqr_primary_pool",
            primary_retrieval_pipeline=primary,
            complementary_retrieval_pipeline=complementary,
            candidate_pool_mode="primary",
            n_steps=5,
        )
        with (
            patch.object(pipeline, "_get_query_embedding_by_id", return_value=None),
            patch.object(pipeline, "_get_candidate_embeddings", return_value=([], np.empty((0, 0)))),
        ):
            results = await pipeline._retrieve_by_id(query_id=1, top_k=3)

        result_doc_ids = [result["doc_id"] for result in results]
        assert set(result_doc_ids) == {1, 2}
        assert 3 not in result_doc_ids

    @pytest.mark.asyncio
    async def test_candidate_pool_mode_union_includes_complementary_candidates(
        self,
        session_factory: sessionmaker[Session],
    ):
        primary = _make_stub_pipeline(
            "vector_search",
            by_id_results=[{"doc_id": 1, "score": 0.9}, {"doc_id": 2, "score": 0.8}],
            by_text_results=[],
        )
        complementary = _make_stub_pipeline(
            "bm25",
            by_id_results=[{"doc_id": 3, "score": 10.0}],
            by_text_results=[],
        )
        pipeline = GQRHybridRetrievalPipeline(
            session_factory=session_factory,
            name="gqr_union_pool",
            primary_retrieval_pipeline=primary,
            complementary_retrieval_pipeline=complementary,
            candidate_pool_mode="union",
            n_steps=5,
        )
        with (
            patch.object(pipeline, "_get_query_embedding_by_id", return_value=None),
            patch.object(pipeline, "_get_candidate_embeddings", return_value=([], np.empty((0, 0)))),
        ):
            results = await pipeline._retrieve_by_id(query_id=1, top_k=3)

        result_doc_ids = [result["doc_id"] for result in results]
        assert 3 in result_doc_ids

    @pytest.mark.asyncio
    async def test_embedding_level_refinement_reranks_with_guidance(
        self,
        session_factory: sessionmaker[Session],
    ):
        primary = _make_stub_pipeline(
            "vector_search",
            by_id_results=[{"doc_id": 1, "score": 0.9}, {"doc_id": 2, "score": 0.1}],
            by_text_results=[],
        )
        complementary = _make_stub_pipeline(
            "bm25",
            by_id_results=[{"doc_id": 2, "score": 10.0}, {"doc_id": 1, "score": 0.1}],
            by_text_results=[],
        )
        pipeline = GQRHybridRetrievalPipeline(
            session_factory=session_factory,
            name="gqr_embedding_refinement",
            primary_retrieval_pipeline=primary,
            complementary_retrieval_pipeline=complementary,
            candidate_pool_mode="primary",
            mixture_alpha=1.0,
            n_steps=50,
            learning_rate=0.5,
            temperature=0.5,
        )
        with (
            patch.object(pipeline, "_get_query_embedding_by_id", return_value=np.asarray([1.0, 0.0])),
            patch.object(
                pipeline,
                "_get_candidate_embeddings",
                return_value=([1, 2], np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)),
            ),
        ):
            results = await pipeline._retrieve_by_id(query_id=1, top_k=2)

        assert results[0]["doc_id"] == 2

    @pytest.mark.asyncio
    async def test_retrieve_by_text_uses_score_space_when_embedding_model_missing(
        self,
        session_factory: sessionmaker[Session],
    ):
        primary = _make_stub_pipeline(
            "vector_search",
            by_id_results=[],
            by_text_results=[{"doc_id": 1, "score": 0.9}, {"doc_id": 2, "score": 0.1}],
        )
        complementary = _make_stub_pipeline(
            "bm25",
            by_id_results=[],
            by_text_results=[{"doc_id": 2, "score": 8.0}, {"doc_id": 1, "score": 0.2}],
        )
        pipeline = GQRHybridRetrievalPipeline(
            session_factory=session_factory,
            name="gqr_text_score_space",
            primary_retrieval_pipeline=primary,
            complementary_retrieval_pipeline=complementary,
            candidate_pool_mode="primary",
            n_steps=10,
        )
        with patch.object(pipeline, "_get_candidate_embeddings", return_value=([], np.empty((0, 0)))):
            results = await pipeline._retrieve_by_text("ad hoc query", top_k=2)

        assert len(results) == 2
        assert {results[0]["doc_id"], results[1]["doc_id"]} == {1, 2}
