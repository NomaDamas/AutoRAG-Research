"""Test cases for VectorSearchRetrievalPipeline.

Tests the vector search retrieval pipeline logic using mocked vector search.
Supports single-vector and multi-vector (MaxSim) search modes using pre-computed embeddings.
"""

from unittest.mock import patch

import pytest
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.orm.repository.chunk_retrieved_result import ChunkRetrievedResultRepository
from autorag_research.pipelines.retrieval.vector_search import (
    VectorSearchPipelineConfig,
    VectorSearchRetrievalPipeline,
)
from tests.autorag_research.pipelines.pipeline_test_utils import (
    PipelineTestConfig,
    PipelineTestVerifier,
)


class TestVectorSearchRetrievalPipeline:
    """Tests for VectorSearchRetrievalPipeline."""

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

    def test_pipeline_creation_single_mode(
        self,
        session_factory: sessionmaker[Session],
        cleanup_pipeline_results: list[int],
    ):
        """Test that pipeline is created correctly with single search mode."""
        pipeline = VectorSearchRetrievalPipeline(
            session_factory=session_factory,
            name="test_vector_search_single",
            search_mode="single",
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        assert pipeline.pipeline_id > 0
        assert pipeline.search_mode == "single"

    def test_pipeline_creation_multi_mode(
        self,
        session_factory: sessionmaker[Session],
        cleanup_pipeline_results: list[int],
    ):
        """Test that pipeline is created correctly with multi search mode."""

        pipeline = VectorSearchRetrievalPipeline(
            session_factory=session_factory,
            name="test_vector_search_multi",
            search_mode="multi",
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        assert pipeline.pipeline_id > 0
        assert pipeline.search_mode == "multi"

    def test_pipeline_creation_default_mode(
        self,
        session_factory: sessionmaker[Session],
        cleanup_pipeline_results: list[int],
    ):
        """Test that pipeline uses single mode by default."""
        pipeline = VectorSearchRetrievalPipeline(
            session_factory=session_factory,
            name="test_vector_search_default",
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        assert pipeline.search_mode == "single"

    def test_pipeline_config_single_mode(
        self,
        session_factory: sessionmaker[Session],
        cleanup_pipeline_results: list[int],
    ):
        """Test that pipeline config is correct for single search mode."""
        pipeline = VectorSearchRetrievalPipeline(
            session_factory=session_factory,
            name="test_vector_search_config_single",
            search_mode="single",
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        config = pipeline._get_pipeline_config()
        assert config["type"] == "vector_search"
        assert config["search_mode"] == "single"

    def test_pipeline_config_multi_mode(
        self,
        session_factory: sessionmaker[Session],
        cleanup_pipeline_results: list[int],
    ):
        """Test that pipeline config is correct for multi search mode."""
        pipeline = VectorSearchRetrievalPipeline(
            session_factory=session_factory,
            name="test_vector_search_config_multi",
            search_mode="multi",
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        config = pipeline._get_pipeline_config()
        assert config["type"] == "vector_search"
        assert config["search_mode"] == "multi"

    def test_run_full_pipeline(
        self,
        session_factory: sessionmaker[Session],
        cleanup_pipeline_results: list[int],
    ):
        """Test running the full pipeline with mocked vector search."""
        from autorag_research.orm.repository.query import QueryRepository

        # Count actual queries in database
        session = session_factory()
        try:
            query_repo = QueryRepository(session)
            query_count = query_repo.count()
        finally:
            session.close()

        # Mock service results - return results for each query
        mock_result = [
            {"doc_id": 1, "score": 0.9, "content": "Content 1"},
            {"doc_id": 2, "score": 0.8, "content": "Content 2"},
        ]

        def mock_vector_search(query_ids, top_k, search_mode="single"):
            """Return mock results for each query ID."""
            return [mock_result for _ in query_ids]

        with patch(
            "autorag_research.orm.service.retrieval_pipeline.RetrievalPipelineService.vector_search"
        ) as mock_search:
            mock_search.side_effect = mock_vector_search

            pipeline = VectorSearchRetrievalPipeline(
                session_factory=session_factory,
                name="test_vector_search_full_run",
            )
            cleanup_pipeline_results.append(pipeline.pipeline_id)

            result = pipeline.run(top_k=3)

            # Verify using test utilities
            config = PipelineTestConfig(
                pipeline_type="retrieval",
                expected_total_queries=query_count,
                expected_min_results=0,
                check_persistence=True,
            )
            verifier = PipelineTestVerifier(result, pipeline.pipeline_id, session_factory, config)
            verifier.verify_all()


class TestVectorSearchPipelineConfig:
    """Tests for VectorSearchPipelineConfig dataclass."""

    def test_config_get_pipeline_class(self):
        """Test that config returns correct pipeline class."""

        config = VectorSearchPipelineConfig(
            name="test_config",
            search_mode="single",
        )

        assert config.get_pipeline_class() == VectorSearchRetrievalPipeline

    def test_config_get_pipeline_kwargs_single(self):
        """Test that config returns correct pipeline kwargs for single mode."""
        config = VectorSearchPipelineConfig(
            name="test_config",
            search_mode="single",
        )

        kwargs = config.get_pipeline_kwargs()

        assert kwargs["search_mode"] == "single"

    def test_config_get_pipeline_kwargs_multi(self):
        """Test that config returns correct pipeline kwargs for multi mode."""
        config = VectorSearchPipelineConfig(
            name="test_config",
            search_mode="multi",
        )

        kwargs = config.get_pipeline_kwargs()

        assert kwargs["search_mode"] == "multi"

    def test_config_default_search_mode(self):
        """Test that config uses single search mode by default."""
        config = VectorSearchPipelineConfig(
            name="test_config",
        )

        assert config.search_mode == "single"
