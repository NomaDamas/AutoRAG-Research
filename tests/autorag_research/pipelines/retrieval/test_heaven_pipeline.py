"""Tests for the HEAVEN retrieval pipeline."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.orm.repository.image_chunk_retrieved_result import ImageChunkRetrievedResultRepository
from autorag_research.orm.repository.query import QueryRepository
from autorag_research.orm.service.retrieval_pipeline import RetrievalPipelineService
from autorag_research.pipelines.retrieval.heaven import (
    HEAVENPipelineConfig,
    HEAVENRetrievalPipeline,
    _combine_heaven_scores,
    _estimate_key_vector_count,
)


class TestHEAVENHelpers:
    """Unit tests for HEAVEN helper functions."""

    def test_estimate_key_vector_count_uses_nouns_and_preserves_one(self):
        """Noun-heavy queries should keep a proportional vector budget with a floor of one."""
        with patch(
            "autorag_research.pipelines.retrieval.heaven.nltk.pos_tag",
            side_effect=[
                [("invoice", "NN"), ("status", "NN"), ("for", "IN"), ("order", "NN")],
                [("quickly", "RB"), ("please", "UH")],
            ],
        ):
            assert (
                _estimate_key_vector_count("invoice status for order", total_query_vectors=8, default_keep_ratio=0.25)
                == 6
            )
            assert _estimate_key_vector_count("quickly please", total_query_vectors=3, default_keep_ratio=0.2) == 1

    def test_combine_heaven_scores_applies_stage2_refinement(self):
        """Only refined candidates should receive the non-key stage contribution."""
        stage1_results = [
            {"doc_id": 1, "score": 0.9},
            {"doc_id": 2, "score": 0.8},
            {"doc_id": 3, "score": 0.7},
        ]
        key_scores = {1: 0.95, 2: 0.7, 3: 0.2}
        non_key_scores = {1: 0.05, 2: 0.9}

        combined = _combine_heaven_scores(
            stage1_results=stage1_results,
            key_scores=key_scores,
            non_key_scores=non_key_scores,
            refine_count=2,
            stage1_weight=0.25,
            top_k=3,
        )

        assert [result["doc_id"] for result in combined] == [2, 1, 3]
        assert combined[2]["score"] < combined[1]["score"]


class TestHEAVENPipeline:
    """Tests for HEAVENRetrievalPipeline."""

    @pytest.fixture
    def cleanup_pipeline_results(self, session_factory: sessionmaker[Session]):
        """Cleanup fixture that deletes HEAVEN pipeline results through the service API."""
        created_pipeline_ids: list[int] = []

        yield created_pipeline_ids

        service = RetrievalPipelineService(session_factory)
        for pipeline_id in created_pipeline_ids:
            service.delete_pipeline_results(pipeline_id)

    def test_pipeline_config_getters(self):
        """Config should resolve to the HEAVEN pipeline and preserve stage parameters."""
        config = HEAVENPipelineConfig(
            name="heaven_cfg",
            stage1_candidate_count=64,
            stage2_refine_ratio=0.5,
            stage1_weight=0.4,
            default_key_token_ratio=0.3,
        )

        assert config.get_pipeline_class() is HEAVENRetrievalPipeline
        assert config.get_pipeline_kwargs() == {
            "stage1_candidate_count": 64,
            "stage2_refine_ratio": 0.5,
            "stage1_weight": 0.4,
            "default_key_token_ratio": 0.3,
            "single_vector_embedding_model": None,
            "multi_vector_embedding_model": None,
        }

    def test_retrieve_by_id_uses_stage1_candidates_then_refines_with_stage2(
        self,
        session_factory: sessionmaker[Session],
        cleanup_pipeline_results: list[int],
    ):
        """HEAVEN should rerank stage1 image candidates with stage2 scores."""
        pipeline = HEAVENRetrievalPipeline(
            session_factory=session_factory,
            name="test_heaven_retrieve",
            stage1_candidate_count=3,
            stage2_refine_ratio=0.5,
            stage1_weight=0.25,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        with (
            patch.object(
                pipeline,
                "_fetch_stored_query",
                return_value=(
                    "invoice status for order",
                    [1.0, 0.0],
                    [[1.0, 0.0], [0.9, 0.1], [0.2, 0.8], [0.1, 0.9]],
                ),
            ),
            patch.object(
                pipeline,
                "_run_stage1_search_from_embedding",
                return_value=[
                    {"doc_id": 1, "score": 0.9},
                    {"doc_id": 2, "score": 0.8},
                    {"doc_id": 3, "score": 0.7},
                ],
            ),
            patch.object(
                pipeline,
                "_fetch_candidate_multi_embeddings",
                return_value={1: [[1.0, 0.0]], 2: [[0.8, 0.2]], 3: [[0.1, 0.9]]},
            ),
            patch("autorag_research.pipelines.retrieval.heaven._estimate_key_vector_count", return_value=2),
            patch.object(
                pipeline,
                "_score_candidates",
                side_effect=[
                    {1: 1.0, 2: 0.7, 3: 0.2},
                    {1: 0.2, 2: 0.9},
                ],
            ) as mock_score,
        ):
            results = asyncio.run(pipeline._retrieve_by_id(query_id=1, top_k=2))

        assert [result["doc_id"] for result in results] == [2, 1]
        assert list(mock_score.call_args_list[1].args[1].keys()) == [1, 2]

    def test_run_persists_image_chunk_results(
        self,
        session_factory: sessionmaker[Session],
        cleanup_pipeline_results: list[int],
    ):
        """Batch runs should persist ImageChunkRetrievedResult rows instead of text chunk results."""
        session = session_factory()
        try:
            query_count = QueryRepository(session).count()
        finally:
            session.close()

        pipeline = HEAVENRetrievalPipeline(
            session_factory=session_factory,
            name="test_heaven_run",
            stage1_candidate_count=4,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        mock_results = [
            {"doc_id": 1, "score": 0.95},
            {"doc_id": 2, "score": 0.75},
        ]

        with patch.object(pipeline, "_retrieve_by_id", new=AsyncMock(return_value=mock_results)):
            result = pipeline.run(top_k=2)

        session = session_factory()
        try:
            stored = ImageChunkRetrievedResultRepository(session).get_by_pipeline(pipeline.pipeline_id)
        finally:
            session.close()

        assert result["pipeline_id"] == pipeline.pipeline_id
        assert result["total_queries"] == query_count
        assert result["total_results"] == query_count * len(mock_results)
        assert len(stored) == query_count * len(mock_results)
