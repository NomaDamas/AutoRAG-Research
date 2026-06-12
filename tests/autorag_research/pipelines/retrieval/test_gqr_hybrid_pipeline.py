"""Tests for GQR hybrid retrieval pipeline."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.exceptions import EmbeddingError
from autorag_research.pipelines.retrieval.base import BaseRetrievalPipeline
from autorag_research.pipelines.retrieval.gqr_hybrid import (
    GQRHybridRetrievalPipeline,
    GQRHybridRetrievalPipelineConfig,
    _maxsim_gradients,
    _maxsim_scores,
)


class StubRetrievalPipeline(BaseRetrievalPipeline):
    """Typed retrieval test double for GQR child pipelines."""

    def __init__(
        self,
        name: str,
        by_id_results: list[dict],
        by_text_results: list[dict] | BaseException,
    ):
        self.name = name
        self.pipeline_id = 1
        self._retrieve_by_id_mock = AsyncMock(return_value=by_id_results)
        self._retrieve_by_text_mock = AsyncMock(
            side_effect=by_text_results if isinstance(by_text_results, BaseException) else None
        )
        if not isinstance(by_text_results, BaseException):
            self._retrieve_by_text_mock.return_value = by_text_results

    def _get_pipeline_config(self) -> dict:
        return {"type": "stub"}

    async def _retrieve_by_id(self, query_id: int | str, top_k: int) -> list[dict]:
        return await self._retrieve_by_id_mock(query_id, top_k)

    async def _retrieve_by_text(self, query_text: str, top_k: int) -> list[dict]:
        return await self._retrieve_by_text_mock(query_text, top_k)


def _make_stub_pipeline(
    name: str,
    by_id_results: list[dict],
    by_text_results: list[dict] | BaseException,
) -> StubRetrievalPipeline:
    return StubRetrievalPipeline(name=name, by_id_results=by_id_results, by_text_results=by_text_results)


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(p * np.log(p / q)))


class TestMaxSimScoring:
    """Tests for GQR multi-vector MaxSim helpers."""

    def test_maxsim_scores_hand_computed_example(self):
        query_matrix = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        candidates = [
            np.asarray([[1.0, 0.0], [0.5, 0.5]], dtype=np.float64),
            np.asarray([[0.0, 0.25], [0.2, 0.8], [0.4, 0.1]], dtype=np.float64),
        ]

        scores = _maxsim_scores(query_matrix, candidates)

        assert scores.tolist() == pytest.approx([0.75, 0.6])

    def test_maxsim_gradient_matches_finite_difference_away_from_ties(self):
        query_matrix = np.asarray([[0.8, 0.1], [0.2, 0.9]], dtype=np.float64)
        candidates = [np.asarray([[0.9, 0.0], [0.0, 0.7]], dtype=np.float64)]
        analytic_gradient = _maxsim_gradients(query_matrix, candidates)[0]
        numeric_gradient = np.zeros_like(query_matrix)
        epsilon = 1e-6

        for row in range(query_matrix.shape[0]):
            for col in range(query_matrix.shape[1]):
                plus = query_matrix.copy()
                minus = query_matrix.copy()
                plus[row, col] += epsilon
                minus[row, col] -= epsilon
                numeric_gradient[row, col] = (
                    _maxsim_scores(plus, candidates)[0] - _maxsim_scores(minus, candidates)[0]
                ) / (2 * epsilon)

        assert numeric_gradient == pytest.approx(analytic_gradient, abs=1e-5)


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
            "scorer_mode": "auto",
        }

    def test_shipped_yaml_config_instantiates(self):
        config = OmegaConf.load("configs/pipelines/retrieval/gqr_hybrid.yaml")

        instantiated = instantiate(config)

        assert isinstance(instantiated, GQRHybridRetrievalPipelineConfig)
        assert instantiated.primary_retrieval_pipeline_name == "vector_search"
        assert instantiated.complementary_retrieval_pipeline_name == "bm25"
        assert instantiated.candidate_pool_mode == "union"
        assert instantiated.scorer_mode == "auto"


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
    async def test_partial_candidate_embeddings_use_score_space_for_whole_pool(
        self,
        session_factory: sessionmaker[Session],
    ):
        primary = _make_stub_pipeline(
            "vector_search",
            by_id_results=[{"doc_id": 1, "score": 0.9}],
            by_text_results=[],
        )
        complementary = _make_stub_pipeline(
            "bm25",
            by_id_results=[{"doc_id": 2, "score": 8.0}],
            by_text_results=[],
        )
        pipeline = GQRHybridRetrievalPipeline(
            session_factory=session_factory,
            name="gqr_partial_embeddings",
            primary_retrieval_pipeline=primary,
            complementary_retrieval_pipeline=complementary,
            candidate_pool_mode="union",
            n_steps=5,
        )
        with (
            patch.object(pipeline, "_get_query_embedding_by_id", return_value=np.asarray([1.0, 0.0])),
            patch.object(
                pipeline,
                "_get_candidate_embeddings",
                return_value=([1], np.asarray([[1.0, 0.0]], dtype=np.float64)),
            ),
            patch.object(pipeline, "_optimize_query_embedding", side_effect=AssertionError("mixed score spaces")),
            patch.object(pipeline, "_optimize_in_score_space", return_value=np.asarray([0.1, 0.9])),
        ):
            results = await pipeline._retrieve_by_id(query_id=1, top_k=2)

        assert [result["doc_id"] for result in results] == [2, 1]
        assert [result["score"] for result in results] == [0.9, 0.1]

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

    def test_score_space_target_moves_with_primary_distribution_each_step(
        self,
        session_factory: sessionmaker[Session],
    ):
        primary = _make_stub_pipeline("vector_search", by_id_results=[], by_text_results=[])
        complementary = _make_stub_pipeline("bm25", by_id_results=[], by_text_results=[])
        pipeline = GQRHybridRetrievalPipeline(
            session_factory=session_factory,
            name="gqr_per_step_target",
            primary_retrieval_pipeline=primary,
            complementary_retrieval_pipeline=complementary,
            mixture_alpha=0.5,
            n_steps=600,
            learning_rate=1.0,
            temperature=1.0,
        )
        primary_scores = np.asarray([4.0, 0.0], dtype=np.float64)
        complementary_distribution = np.asarray([0.1, 0.9], dtype=np.float64)
        initial_primary_distribution = np.exp(primary_scores) / np.exp(primary_scores).sum()
        fixed_initial_blend = 0.5 * initial_primary_distribution + 0.5 * complementary_distribution

        optimized_logits = pipeline._optimize_in_score_space(primary_scores, complementary_distribution)
        optimized_distribution = np.exp(optimized_logits) / np.exp(optimized_logits).sum()

        assert np.linalg.norm(optimized_distribution - complementary_distribution) < 0.05
        assert np.linalg.norm(optimized_distribution - fixed_initial_blend) > 0.25

    def test_maxsim_optimization_moves_distribution_toward_complementary(
        self,
        session_factory: sessionmaker[Session],
    ):
        primary = _make_stub_pipeline("vector_search", by_id_results=[], by_text_results=[])
        complementary = _make_stub_pipeline("bm25", by_id_results=[], by_text_results=[])
        pipeline = GQRHybridRetrievalPipeline(
            session_factory=session_factory,
            name="gqr_maxsim_optimization",
            primary_retrieval_pipeline=primary,
            complementary_retrieval_pipeline=complementary,
            scorer_mode="multi",
            mixture_alpha=1.0,
            n_steps=300,
            learning_rate=0.8,
            temperature=1.0,
        )
        query_matrix = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        candidates = [
            np.asarray([[1.0, 0.0], [0.0, 0.1]]),
            np.asarray([[0.1, 0.0], [0.0, 1.0]]),
        ]
        complementary_distribution = np.asarray([0.1, 0.9], dtype=np.float64)
        initial_distribution = np.exp(_maxsim_scores(query_matrix, candidates))
        initial_distribution = initial_distribution / initial_distribution.sum()

        optimized_scores = pipeline._optimize_query_multi_embedding(
            query_matrix,
            candidates,
            complementary_distribution,
        )
        optimized_distribution = np.exp(optimized_scores) / np.exp(optimized_scores).sum()

        assert _kl_divergence(complementary_distribution, optimized_distribution) < _kl_divergence(
            complementary_distribution,
            initial_distribution,
        )

    @pytest.mark.asyncio
    async def test_auto_mode_uses_multi_path_for_multi_primary(
        self,
        session_factory: sessionmaker[Session],
    ):
        primary = _make_stub_pipeline(
            "vector_search",
            by_id_results=[{"doc_id": 1, "score": 0.9}, {"doc_id": 2, "score": 0.1}],
            by_text_results=[],
        )
        primary.search_mode = "multi"
        complementary = _make_stub_pipeline(
            "bm25",
            by_id_results=[{"doc_id": 2, "score": 10.0}, {"doc_id": 1, "score": 0.1}],
            by_text_results=[],
        )
        pipeline = GQRHybridRetrievalPipeline(
            session_factory=session_factory,
            name="gqr_auto_multi",
            primary_retrieval_pipeline=primary,
            complementary_retrieval_pipeline=complementary,
            scorer_mode="auto",
            candidate_pool_mode="primary",
        )
        with (
            patch.object(pipeline, "_get_query_embedding_by_id", return_value=None),
            patch.object(
                pipeline,
                "_get_query_multi_embedding_by_id",
                return_value=np.asarray([[1.0, 0.0]]),
            ),
            patch.object(
                pipeline,
                "_get_candidate_multi_embeddings",
                return_value=(
                    [1, 2],
                    [np.asarray([[1.0, 0.0]]), np.asarray([[0.0, 1.0]])],
                ),
            ) as multi_embeddings,
            patch.object(
                pipeline,
                "_optimize_query_multi_embedding",
                return_value=np.asarray([0.2, 0.8]),
            ) as optimize_multi,
            patch.object(
                pipeline,
                "_optimize_in_score_space",
                side_effect=AssertionError("expected MaxSim path"),
            ),
        ):
            results = await pipeline._retrieve_by_id(query_id=1, top_k=2)

        multi_embeddings.assert_called_once_with([1, 2])
        optimize_multi.assert_called_once()
        assert [result["doc_id"] for result in results] == [2, 1]

    @pytest.mark.asyncio
    async def test_multi_mode_missing_candidate_embeddings_falls_back_to_score_space(
        self,
        session_factory: sessionmaker[Session],
    ):
        primary = _make_stub_pipeline(
            "vector_search",
            by_id_results=[{"doc_id": 1, "score": 0.9}, {"doc_id": 2, "score": 0.1}],
            by_text_results=[],
        )
        primary.search_mode = "multi"
        complementary = _make_stub_pipeline(
            "bm25",
            by_id_results=[{"doc_id": 2, "score": 10.0}, {"doc_id": 1, "score": 0.1}],
            by_text_results=[],
        )
        pipeline = GQRHybridRetrievalPipeline(
            session_factory=session_factory,
            name="gqr_multi_missing_embeddings",
            primary_retrieval_pipeline=primary,
            complementary_retrieval_pipeline=complementary,
            scorer_mode="auto",
            candidate_pool_mode="primary",
        )
        with (
            patch.object(pipeline, "_get_query_embedding_by_id", return_value=None),
            patch.object(
                pipeline,
                "_get_query_multi_embedding_by_id",
                return_value=np.asarray([[1.0, 0.0]]),
            ),
            patch.object(
                pipeline,
                "_get_candidate_multi_embeddings",
                return_value=([1], [np.asarray([[1.0, 0.0]])]),
            ),
            patch.object(
                pipeline,
                "_optimize_query_multi_embedding",
                side_effect=AssertionError("partial MaxSim pool"),
            ),
            patch.object(
                pipeline,
                "_optimize_in_score_space",
                return_value=np.asarray([0.1, 0.9]),
            ) as score_space,
        ):
            results = await pipeline._retrieve_by_id(query_id=1, top_k=2)

        score_space.assert_called_once()
        assert [result["doc_id"] for result in results] == [2, 1]

    def test_raw_score_scale_changes_softmax_distribution(
        self,
        session_factory: sessionmaker[Session],
    ):
        primary = _make_stub_pipeline("vector_search", by_id_results=[], by_text_results=[])
        complementary = _make_stub_pipeline("bm25", by_id_results=[], by_text_results=[])
        pipeline = GQRHybridRetrievalPipeline(
            session_factory=session_factory,
            name="gqr_raw_scores",
            primary_retrieval_pipeline=primary,
            complementary_retrieval_pipeline=complementary,
            temperature=1.0,
        )
        low_scale_scores = pipeline._build_score_vector([1, 2], {1: 2.0, 2: 1.0})
        high_scale_scores = pipeline._build_score_vector([1, 2], {1: 20.0, 2: 10.0})

        low_scale_distribution = np.exp(low_scale_scores) / np.exp(low_scale_scores).sum()
        high_scale_distribution = np.exp(high_scale_scores) / np.exp(high_scale_scores).sum()

        assert low_scale_scores.tolist() == [2.0, 1.0]
        assert high_scale_scores.tolist() == [20.0, 10.0]
        assert not np.allclose(low_scale_distribution, high_scale_distribution)
        assert high_scale_distribution[0] > low_scale_distribution[0]

    @pytest.mark.asyncio
    async def test_retrieve_by_text_falls_back_to_complementary_when_primary_needs_embedding(
        self,
        session_factory: sessionmaker[Session],
    ):
        primary = _make_stub_pipeline(
            "vector_search",
            by_id_results=[],
            by_text_results=EmbeddingError(),
        )
        complementary = _make_stub_pipeline(
            "bm25",
            by_id_results=[],
            by_text_results=[{"doc_id": 2, "score": 8.0}, {"doc_id": 1, "score": 0.2}],
        )
        pipeline = GQRHybridRetrievalPipeline(
            session_factory=session_factory,
            name="gqr_text_embedding_boundary",
            primary_retrieval_pipeline=primary,
            complementary_retrieval_pipeline=complementary,
            candidate_pool_mode="union",
            n_steps=10,
        )
        with patch.object(pipeline, "_get_candidate_embeddings", return_value=([], np.empty((0, 0)))):
            results = await pipeline._retrieve_by_text("ad hoc query", top_k=2)

        assert [result["doc_id"] for result in results] == [2, 1]
        primary._retrieve_by_text_mock.assert_awaited_once_with("ad hoc query", 4)
        complementary._retrieve_by_text_mock.assert_awaited_once_with("ad hoc query", 4)
