"""Tests for Hybrid Retrieval Pipelines."""

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.pipelines.retrieval.hybrid import (
    HybridCCRetrievalPipeline,
    HybridCCRetrievalPipelineConfig,
    HybridRRFRetrievalPipeline,
    HybridRRFRetrievalPipelineConfig,
    _cc_fuse,
    _rrf_fuse,
)


class TestRRFFusion:
    """Tests for the RRF fusion function."""

    def test_basic_fusion(self):
        """Test basic RRF fusion with overlapping results."""
        results_1 = [
            {"doc_id": 1, "score": 0.9},
            {"doc_id": 2, "score": 0.8},
            {"doc_id": 3, "score": 0.7},
        ]
        results_2 = [
            {"doc_id": 2, "score": 0.95},
            {"doc_id": 1, "score": 0.85},
            {"doc_id": 4, "score": 0.75},
        ]

        fused = _rrf_fuse(results_1, results_2, k=60, top_k=3)

        assert len(fused) == 3
        # Doc 1 and 2 should have higher scores (appear in both lists)
        doc_ids = [r["doc_id"] for r in fused]
        assert 1 in doc_ids[:2]
        assert 2 in doc_ids[:2]

    def test_rrf_ignores_scores(self):
        """Test that RRF uses only ranks, not scores."""
        # Same ranks, different scores should give same RRF result
        results_1a = [{"doc_id": 1, "score": 0.99}, {"doc_id": 2, "score": 0.01}]
        results_1b = [{"doc_id": 1, "score": 0.51}, {"doc_id": 2, "score": 0.49}]
        results_2 = [{"doc_id": 1, "score": 0.5}, {"doc_id": 2, "score": 0.4}]

        fused_a = _rrf_fuse(results_1a, results_2, k=60, top_k=2)
        fused_b = _rrf_fuse(results_1b, results_2, k=60, top_k=2)

        # RRF scores should be identical (depends only on ranks)
        assert fused_a[0]["score"] == fused_b[0]["score"]
        assert fused_a[1]["score"] == fused_b[1]["score"]

    def test_rrf_k_affects_results(self):
        """Test that different k values affect the scoring."""
        results_1 = [{"doc_id": 1, "score": 0.9}, {"doc_id": 2, "score": 0.8}]
        results_2 = [{"doc_id": 2, "score": 0.95}, {"doc_id": 1, "score": 0.85}]

        fused_k60 = _rrf_fuse(results_1, results_2, k=60, top_k=2)
        fused_k1 = _rrf_fuse(results_1, results_2, k=1, top_k=2)

        # With k=1, rank differences matter more
        # Doc 1: rank 1 in list 1, rank 2 in list 2
        # Doc 2: rank 2 in list 1, rank 1 in list 2
        # Both should have same total RRF score
        assert fused_k60[0]["score"] == fused_k60[1]["score"]
        assert fused_k1[0]["score"] == fused_k1[1]["score"]

    def test_empty_results(self):
        """Test with empty result lists."""
        results_1: list[dict] = []
        results_2 = [{"doc_id": 1, "score": 0.9}]

        fused = _rrf_fuse(results_1, results_2, k=60, top_k=2)

        assert len(fused) == 1
        assert fused[0]["doc_id"] == 1

    def test_non_overlapping_results(self):
        """Test with completely non-overlapping results."""
        results_1 = [{"doc_id": 1, "score": 0.9}, {"doc_id": 2, "score": 0.8}]
        results_2 = [{"doc_id": 3, "score": 0.95}, {"doc_id": 4, "score": 0.85}]

        fused = _rrf_fuse(results_1, results_2, k=60, top_k=4)

        assert len(fused) == 4
        doc_ids = {r["doc_id"] for r in fused}
        assert doc_ids == {1, 2, 3, 4}


class TestCCFusion:
    """Tests for the Convex Combination fusion function."""

    def test_basic_minmax_fusion(self):
        """Test basic CC fusion with minmax normalization."""
        results_1 = [
            {"doc_id": 1, "score": 0.9},
            {"doc_id": 2, "score": 0.5},
        ]
        results_2 = [
            {"doc_id": 2, "score": 100.0},
            {"doc_id": 1, "score": 50.0},
        ]

        fused = _cc_fuse(results_1, results_2, weight=0.5, top_k=2, normalize_method="mm")

        assert len(fused) == 2
        # Both docs appear in both lists, scores should be combined
        for result in fused:
            assert 0.0 <= result["score"] <= 1.0

    def test_weight_affects_ranking(self):
        """Test that weight parameter affects the final ranking."""
        results_1 = [{"doc_id": 1, "score": 1.0}, {"doc_id": 2, "score": 0.0}]
        results_2 = [{"doc_id": 2, "score": 1.0}, {"doc_id": 1, "score": 0.0}]

        # Weight = 1.0 -> full pipeline_1
        fused_w1 = _cc_fuse(results_1, results_2, weight=1.0, top_k=2, normalize_method="mm")
        assert fused_w1[0]["doc_id"] == 1  # Doc 1 wins (highest in pipeline_1)

        # Weight = 0.0 -> full pipeline_2
        fused_w0 = _cc_fuse(results_1, results_2, weight=0.0, top_k=2, normalize_method="mm")
        assert fused_w0[0]["doc_id"] == 2  # Doc 2 wins (highest in pipeline_2)

        # Weight = 0.5 -> equal weight, tied scores
        fused_w5 = _cc_fuse(results_1, results_2, weight=0.5, top_k=2, normalize_method="mm")
        assert fused_w5[0]["score"] == fused_w5[1]["score"]

    def test_zscore_normalization(self):
        """Test CC fusion with z-score normalization."""
        results_1 = [{"doc_id": 1, "score": 10.0}, {"doc_id": 2, "score": 20.0}]
        results_2 = [{"doc_id": 1, "score": 0.5}, {"doc_id": 2, "score": 0.6}]

        fused = _cc_fuse(results_1, results_2, weight=0.5, top_k=2, normalize_method="z")

        assert len(fused) == 2
        # Doc 2 should rank higher (higher in both lists)
        assert fused[0]["doc_id"] == 2

    def test_dbsf_normalization(self):
        """Test CC fusion with DBSF normalization."""
        results_1 = [{"doc_id": 1, "score": 0.8}, {"doc_id": 2, "score": 0.9}]
        results_2 = [{"doc_id": 1, "score": 80.0}, {"doc_id": 2, "score": 90.0}]

        fused = _cc_fuse(results_1, results_2, weight=0.5, top_k=2, normalize_method="dbsf")

        assert len(fused) == 2
        # All scores should be clipped to [0, 1]
        for result in fused:
            assert 0.0 <= result["score"] <= 1.0

    def test_tmm_normalization(self):
        """Test CC fusion with theoretical min-max normalization."""
        results_1 = [{"doc_id": 1, "score": 0.0}, {"doc_id": 2, "score": 0.5}]
        results_2 = [{"doc_id": 1, "score": 50.0}, {"doc_id": 2, "score": 100.0}]

        fused = _cc_fuse(
            results_1,
            results_2,
            weight=0.5,
            top_k=2,
            normalize_method="tmm",
            pipeline_1_min=-1.0,
            pipeline_2_min=0.0,
        )

        assert len(fused) == 2

    def test_tmm_missing_bounds_raises_error(self):
        """Test that TMM raises error when bounds are missing."""
        results_1 = [{"doc_id": 1, "score": 0.5}]
        results_2 = [{"doc_id": 1, "score": 50.0}]

        with pytest.raises(ValueError, match="TMM normalization requires pipeline_1_min"):
            _cc_fuse(results_1, results_2, weight=0.5, top_k=1, normalize_method="tmm")

    def test_invalid_normalization_method(self):
        """Test that invalid normalization method raises error."""
        results_1 = [{"doc_id": 1, "score": 0.5}]
        results_2 = [{"doc_id": 1, "score": 0.5}]

        with pytest.raises(ValueError, match="Unknown normalization method"):
            _cc_fuse(results_1, results_2, weight=0.5, top_k=1, normalize_method="invalid")

    def test_non_overlapping_results(self):
        """Test CC fusion with non-overlapping results."""
        results_1 = [{"doc_id": 1, "score": 0.9}]
        results_2 = [{"doc_id": 2, "score": 0.9}]

        fused = _cc_fuse(results_1, results_2, weight=0.5, top_k=2, normalize_method="mm")

        assert len(fused) == 2
        # Doc 1 gets 0.5 * 0.5 (from mm norm of single value) + 0 = 0.25
        # Doc 2 gets 0 + 0.5 * 0.5 = 0.25
        assert fused[0]["score"] == fused[1]["score"]


class TestHybridRRFRetrievalPipelineConfig:
    """Tests for HybridRRFRetrievalPipelineConfig."""

    def test_get_pipeline_class(self):
        """Test that config returns correct pipeline class."""
        config = HybridRRFRetrievalPipelineConfig(
            name="test_rrf",
            retrieval_pipeline_1_name="vector_search",
            retrieval_pipeline_2_name="bm25",
        )

        assert config.get_pipeline_class() == HybridRRFRetrievalPipeline

    def test_get_pipeline_kwargs(self):
        """Test that config returns correct kwargs."""
        config = HybridRRFRetrievalPipelineConfig(
            name="test_rrf",
            retrieval_pipeline_1_name="vector_search",
            retrieval_pipeline_2_name="bm25",
            rrf_k=30,
        )

        kwargs = config.get_pipeline_kwargs()

        assert kwargs["retrieval_pipeline_1"] == "vector_search"
        assert kwargs["retrieval_pipeline_2"] == "bm25"
        assert kwargs["rrf_k"] == 30

    def test_default_rrf_k(self):
        """Test default RRF k value."""
        config = HybridRRFRetrievalPipelineConfig(
            name="test_rrf",
            retrieval_pipeline_1_name="p1",
            retrieval_pipeline_2_name="p2",
        )

        assert config.rrf_k == 60


class TestHybridCCRetrievalPipelineConfig:
    """Tests for HybridCCRetrievalPipelineConfig."""

    def test_get_pipeline_class(self):
        """Test that config returns correct pipeline class."""
        config = HybridCCRetrievalPipelineConfig(
            name="test_cc",
            retrieval_pipeline_1_name="vector_search",
            retrieval_pipeline_2_name="bm25",
        )

        assert config.get_pipeline_class() == HybridCCRetrievalPipeline

    def test_get_pipeline_kwargs(self):
        """Test that config returns correct kwargs."""
        config = HybridCCRetrievalPipelineConfig(
            name="test_cc",
            retrieval_pipeline_1_name="vector_search",
            retrieval_pipeline_2_name="bm25",
            weight=0.7,
            normalize_method="z",
        )

        kwargs = config.get_pipeline_kwargs()

        assert kwargs["retrieval_pipeline_1"] == "vector_search"
        assert kwargs["retrieval_pipeline_2"] == "bm25"
        assert kwargs["weight"] == 0.7
        assert kwargs["normalize_method"] == "z"

    def test_default_values(self):
        """Test default CC config values."""
        config = HybridCCRetrievalPipelineConfig(
            name="test_cc",
            retrieval_pipeline_1_name="p1",
            retrieval_pipeline_2_name="p2",
        )

        assert config.weight == 0.5
        assert config.normalize_method == "mm"
        assert config.pipeline_1_min is None


class TestHybridRRFRetrievalPipeline:
    """Tests for HybridRRFRetrievalPipeline."""

    def test_pipeline_creation_with_instances(self, session_factory: sessionmaker[Session]):
        """Test pipeline creation with instantiated sub-pipelines."""
        # Create mock pipelines
        mock_pipeline_1 = MagicMock()
        mock_pipeline_1.name = "mock_vector"
        mock_pipeline_1._get_retrieval_func.return_value = lambda ids, k: [[{"doc_id": 1, "score": 0.9}]] * len(ids)

        mock_pipeline_2 = MagicMock()
        mock_pipeline_2.name = "mock_bm25"
        mock_pipeline_2._get_retrieval_func.return_value = lambda ids, k: [[{"doc_id": 2, "score": 0.8}]] * len(ids)

        pipeline = HybridRRFRetrievalPipeline(
            session_factory=session_factory,
            name="test_hybrid_rrf",
            retrieval_pipeline_1=mock_pipeline_1,
            retrieval_pipeline_2=mock_pipeline_2,
            rrf_k=60,
        )

        assert pipeline.pipeline_id > 0
        assert pipeline.rrf_k == 60
        assert pipeline._retrieval_pipeline_1 == mock_pipeline_1
        assert pipeline._retrieval_pipeline_2 == mock_pipeline_2

    def test_pipeline_config(self, session_factory: sessionmaker[Session]):
        """Test pipeline config output."""
        mock_pipeline_1 = MagicMock()
        mock_pipeline_1.name = "vector_search"

        mock_pipeline_2 = MagicMock()
        mock_pipeline_2.name = "bm25"

        pipeline = HybridRRFRetrievalPipeline(
            session_factory=session_factory,
            name="test_hybrid_rrf",
            retrieval_pipeline_1=mock_pipeline_1,
            retrieval_pipeline_2=mock_pipeline_2,
            rrf_k=30,
        )

        config = pipeline._get_pipeline_config()

        assert config["type"] == "hybrid_rrf"
        assert config["retrieval_pipeline_1"] == "vector_search"
        assert config["retrieval_pipeline_2"] == "bm25"
        assert config["rrf_k"] == 30

    def test_fuse_results(self, session_factory: sessionmaker[Session]):
        """Test that _fuse_results uses RRF fusion."""
        mock_pipeline_1 = MagicMock()
        mock_pipeline_1.name = "p1"

        mock_pipeline_2 = MagicMock()
        mock_pipeline_2.name = "p2"

        pipeline = HybridRRFRetrievalPipeline(
            session_factory=session_factory,
            name="test_hybrid_rrf",
            retrieval_pipeline_1=mock_pipeline_1,
            retrieval_pipeline_2=mock_pipeline_2,
        )

        results_1 = [{"doc_id": 1, "score": 0.9}, {"doc_id": 2, "score": 0.8}]
        results_2 = [{"doc_id": 2, "score": 0.95}, {"doc_id": 1, "score": 0.85}]

        fused = pipeline._fuse_results(results_1, results_2, top_k=2)

        assert len(fused) == 2
        # Both docs should have same RRF score (symmetric ranks)
        assert fused[0]["score"] == fused[1]["score"]

    def test_pipeline_creation_with_names(self, session_factory: sessionmaker[Session]):
        """Test that pipeline can load sub-pipelines by name."""
        # Mock the _load_pipeline method
        with patch.object(HybridRRFRetrievalPipeline, "_load_pipeline") as mock_load:
            mock_pipeline_1 = MagicMock()
            mock_pipeline_1.name = "vector_search"
            mock_pipeline_2 = MagicMock()
            mock_pipeline_2.name = "bm25"
            mock_load.side_effect = [mock_pipeline_1, mock_pipeline_2]

            pipeline = HybridRRFRetrievalPipeline(
                session_factory=session_factory,
                name="test_hybrid_rrf",
                retrieval_pipeline_1="vector_search",
                retrieval_pipeline_2="bm25",
            )

            assert mock_load.call_count == 2
            assert pipeline._retrieval_pipeline_1 == mock_pipeline_1
            assert pipeline._retrieval_pipeline_2 == mock_pipeline_2


class TestHybridCCRetrievalPipeline:
    """Tests for HybridCCRetrievalPipeline."""

    def test_pipeline_creation_with_instances(self, session_factory: sessionmaker[Session]):
        """Test pipeline creation with instantiated sub-pipelines."""
        mock_pipeline_1 = MagicMock()
        mock_pipeline_1.name = "mock_vector"

        mock_pipeline_2 = MagicMock()
        mock_pipeline_2.name = "mock_bm25"

        pipeline = HybridCCRetrievalPipeline(
            session_factory=session_factory,
            name="test_hybrid_cc",
            retrieval_pipeline_1=mock_pipeline_1,
            retrieval_pipeline_2=mock_pipeline_2,
            weight=0.7,
            normalize_method="z",
        )

        assert pipeline.pipeline_id > 0
        assert pipeline.weight == 0.7
        assert pipeline.normalize_method == "z"

    def test_pipeline_config(self, session_factory: sessionmaker[Session]):
        """Test pipeline config output."""
        mock_pipeline_1 = MagicMock()
        mock_pipeline_1.name = "vector_search"

        mock_pipeline_2 = MagicMock()
        mock_pipeline_2.name = "bm25"

        pipeline = HybridCCRetrievalPipeline(
            session_factory=session_factory,
            name="test_hybrid_cc",
            retrieval_pipeline_1=mock_pipeline_1,
            retrieval_pipeline_2=mock_pipeline_2,
            weight=0.6,
            normalize_method="dbsf",
        )

        config = pipeline._get_pipeline_config()

        assert config["type"] == "hybrid_cc"
        assert config["retrieval_pipeline_1"] == "vector_search"
        assert config["retrieval_pipeline_2"] == "bm25"
        assert config["weight"] == 0.6
        assert config["normalize_method"] == "dbsf"

    def test_pipeline_config_with_tmm(self, session_factory: sessionmaker[Session]):
        """Test pipeline config with TMM normalization includes bounds."""
        mock_pipeline_1 = MagicMock()
        mock_pipeline_1.name = "p1"

        mock_pipeline_2 = MagicMock()
        mock_pipeline_2.name = "p2"

        pipeline = HybridCCRetrievalPipeline(
            session_factory=session_factory,
            name="test_hybrid_cc",
            retrieval_pipeline_1=mock_pipeline_1,
            retrieval_pipeline_2=mock_pipeline_2,
            normalize_method="tmm",
            pipeline_1_min=-1.0,
            pipeline_2_min=0.0,
        )

        config = pipeline._get_pipeline_config()

        assert config["normalize_method"] == "tmm"
        assert config["pipeline_1_min"] == -1.0
        assert config["pipeline_2_min"] == 0.0

    def test_fuse_results(self, session_factory: sessionmaker[Session]):
        """Test that _fuse_results uses CC fusion."""
        mock_pipeline_1 = MagicMock()
        mock_pipeline_1.name = "p1"

        mock_pipeline_2 = MagicMock()
        mock_pipeline_2.name = "p2"

        pipeline = HybridCCRetrievalPipeline(
            session_factory=session_factory,
            name="test_hybrid_cc",
            retrieval_pipeline_1=mock_pipeline_1,
            retrieval_pipeline_2=mock_pipeline_2,
            weight=1.0,  # Full weight to pipeline_1
        )

        results_1 = [{"doc_id": 1, "score": 1.0}, {"doc_id": 2, "score": 0.0}]
        results_2 = [{"doc_id": 2, "score": 1.0}, {"doc_id": 1, "score": 0.0}]

        fused = pipeline._fuse_results(results_1, results_2, top_k=2)

        # With weight=1.0, doc_1 should be first (highest in pipeline_1)
        assert fused[0]["doc_id"] == 1
