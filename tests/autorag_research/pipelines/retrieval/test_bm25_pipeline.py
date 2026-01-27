"""Test cases for BM25RetrievalPipeline.

Tests the BM25 retrieval pipeline logic using mocked BM25 search.
"""

from unittest.mock import patch

import pytest
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.orm.repository.chunk_retrieved_result import ChunkRetrievedResultRepository
from autorag_research.orm.schema import Chunk
from autorag_research.pipelines.retrieval.bm25 import BM25RetrievalPipeline
from tests.autorag_research.pipelines.pipeline_test_utils import (
    PipelineTestConfig,
    PipelineTestVerifier,
)


class TestBM25RetrievalPipeline:
    """Tests for BM25RetrievalPipeline."""

    @pytest.fixture
    def cleanup_pipeline_results(self, session_factory: sessionmaker[Session]):
        """Cleanup fixture that deletes pipeline results after test."""
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

    def test_pipeline_creation(
        self,
        session_factory: sessionmaker[Session],
        cleanup_pipeline_results: list[int],
    ):
        """Test that pipeline is created correctly."""
        pipeline = BM25RetrievalPipeline(
            session_factory=session_factory,
            name="test_bm25_pipeline",
            tokenizer="bert",
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        assert pipeline.pipeline_id > 0
        assert pipeline.tokenizer == "bert"
        assert pipeline.index_name == "idx_chunk_bm25"

    def test_pipeline_config(
        self,
        session_factory: sessionmaker[Session],
        cleanup_pipeline_results: list[int],
    ):
        """Test that pipeline config is correct."""
        pipeline = BM25RetrievalPipeline(
            session_factory=session_factory,
            name="test_bm25_pipeline_config",
            tokenizer="bert",
            index_name="custom_index",
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        config = pipeline._get_pipeline_config()
        assert config["type"] == "bm25"
        assert config["tokenizer"] == "bert"
        assert config["index_name"] == "custom_index"

    def test_retrieve_single_query(
        self,
        session_factory: sessionmaker[Session],
        cleanup_pipeline_results: list[int],
    ):
        """Test single query retrieval with mocked BM25 search."""
        # Use actual Chunk model instances for mock results
        mock_results = [
            (Chunk(id=1, contents="Machine learning content"), 0.95),
            (Chunk(id=2, contents="Deep learning content"), 0.85),
            (Chunk(id=3, contents="Neural network content"), 0.75),
        ]

        with patch("autorag_research.orm.repository.chunk.ChunkRepository.bm25_search") as mock_search:
            mock_search.return_value = mock_results

            pipeline = BM25RetrievalPipeline(
                session_factory=session_factory,
                name="test_bm25_retrieve",
                tokenizer="bert",
            )
            cleanup_pipeline_results.append(pipeline.pipeline_id)

            results = pipeline.retrieve("machine learning", top_k=3)

            assert isinstance(results, list)
            assert len(results) == 3
            for i, result in enumerate(results):
                assert result["doc_id"] == mock_results[i][0].id
                assert result["score"] == mock_results[i][1]

            # Verify search was called with correct parameters
            mock_search.assert_called_once()
            call_kwargs = mock_search.call_args[1]
            assert call_kwargs["query_text"] == "machine learning"
            assert call_kwargs["limit"] == 3
            assert call_kwargs["tokenizer"] == "bert"

    def test_run_full_pipeline(
        self,
        session_factory: sessionmaker[Session],
        cleanup_pipeline_results: list[int],
    ):
        """Test running the full pipeline with mocked BM25 search."""
        # Use actual Chunk model instances for mock results
        mock_results = [
            (Chunk(id=1, contents="Content 1"), 0.9),
            (Chunk(id=2, contents="Content 2"), 0.8),
        ]

        with patch("autorag_research.orm.repository.chunk.ChunkRepository.bm25_search") as mock_search:
            mock_search.return_value = mock_results

            pipeline = BM25RetrievalPipeline(
                session_factory=session_factory,
                name="test_bm25_full_run",
                tokenizer="bert",
            )
            cleanup_pipeline_results.append(pipeline.pipeline_id)

            result = pipeline.run(top_k=3)

            # Verify using test utilities
            config = PipelineTestConfig(
                pipeline_type="retrieval",
                expected_total_queries=5,  # Seed data has 5 queries
                expected_min_results=0,
                check_persistence=True,
            )
            verifier = PipelineTestVerifier(result, pipeline.pipeline_id, session_factory, config)
            verifier.verify_all()

            # Verify search was called for each query
            assert mock_search.call_count == 5

    def test_results_persisted_correctly(
        self,
        session_factory: sessionmaker[Session],
        cleanup_pipeline_results: list[int],
    ):
        """Test that results are correctly persisted in database."""
        # Use actual Chunk model instances for mock results
        mock_results = [
            (Chunk(id=1, contents="Content 1"), 0.95),
            (Chunk(id=2, contents="Content 2"), 0.85),
        ]

        with patch("autorag_research.orm.repository.chunk.ChunkRepository.bm25_search") as mock_search:
            mock_search.return_value = mock_results

            pipeline = BM25RetrievalPipeline(
                session_factory=session_factory,
                name="test_bm25_persistence",
                tokenizer="bert",
            )
            cleanup_pipeline_results.append(pipeline.pipeline_id)

            pipeline.run(top_k=3)

            # Verify persistence
            session = session_factory()
            try:
                repo = ChunkRetrievedResultRepository(session)
                results = repo.get_by_pipeline(pipeline.pipeline_id)

                # Should have results persisted (5 queries * 2 results each = 10)
                assert len(results) == 10

                # All results should have valid scores
                for r in results:
                    assert r.rel_score >= 0
            finally:
                session.close()
