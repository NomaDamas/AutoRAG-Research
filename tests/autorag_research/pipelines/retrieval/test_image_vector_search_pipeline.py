"""Test cases for ImageVectorSearchRetrievalPipeline.

Tests the image vector search retrieval pipeline logic.
Supports single-vector and multi-vector (MaxSim) search modes for image_chunk rows.
"""

import pytest
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.orm.repository.image_chunk_retrieved_result import ImageChunkRetrievedResultRepository
from autorag_research.pipelines.retrieval.image_vector_search import (
    ImageVectorSearchPipelineConfig,
    ImageVectorSearchRetrievalPipeline,
)


class TestImageVectorSearchRetrievalPipeline:
    """Tests for ImageVectorSearchRetrievalPipeline."""

    @pytest.fixture
    def cleanup_pipeline_results(self, session_factory: sessionmaker[Session]):
        """Cleanup fixture that deletes pipeline results after test."""
        created_pipeline_ids: list[int] = []

        yield created_pipeline_ids

        session = session_factory()
        try:
            result_repo = ImageChunkRetrievedResultRepository(session)
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
        pipeline = ImageVectorSearchRetrievalPipeline(
            session_factory=session_factory,
            name="test_image_vector_search_single",
            search_mode="single",
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        assert pipeline.pipeline_id > 0
        assert pipeline.search_mode == "single"
        assert pipeline.retrieval_unit == "image_chunk"

    def test_pipeline_creation_multi_mode(
        self,
        session_factory: sessionmaker[Session],
        cleanup_pipeline_results: list[int],
    ):
        """Test that pipeline is created correctly with multi search mode."""
        pipeline = ImageVectorSearchRetrievalPipeline(
            session_factory=session_factory,
            name="test_image_vector_search_multi",
            search_mode="multi",
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        assert pipeline.pipeline_id > 0
        assert pipeline.search_mode == "multi"
        assert pipeline.retrieval_unit == "image_chunk"

    def test_pipeline_creation_default_mode(
        self,
        session_factory: sessionmaker[Session],
        cleanup_pipeline_results: list[int],
    ):
        """Test that pipeline uses multi mode by default."""
        pipeline = ImageVectorSearchRetrievalPipeline(
            session_factory=session_factory,
            name="test_image_vector_search_default",
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        assert pipeline.search_mode == "multi"
        assert pipeline.retrieval_unit == "image_chunk"

    def test_pipeline_config_single_mode(
        self,
        session_factory: sessionmaker[Session],
        cleanup_pipeline_results: list[int],
    ):
        """Test that pipeline config is correct for single search mode."""
        pipeline = ImageVectorSearchRetrievalPipeline(
            session_factory=session_factory,
            name="test_image_vector_search_config_single",
            search_mode="single",
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        config = pipeline._get_pipeline_config()
        assert config["type"] == "image_vector_search"
        assert config["retrieval_unit"] == "image_chunk"
        assert config["search_mode"] == "single"

    def test_pipeline_config_multi_mode(
        self,
        session_factory: sessionmaker[Session],
        cleanup_pipeline_results: list[int],
    ):
        """Test that pipeline config is correct for multi search mode."""
        pipeline = ImageVectorSearchRetrievalPipeline(
            session_factory=session_factory,
            name="test_image_vector_search_config_multi",
            search_mode="multi",
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        config = pipeline._get_pipeline_config()
        assert config["type"] == "image_vector_search"
        assert config["retrieval_unit"] == "image_chunk"
        assert config["search_mode"] == "multi"


class TestImageVectorSearchPipelineConfig:
    """Tests for ImageVectorSearchPipelineConfig dataclass."""

    def test_config_get_pipeline_class(self):
        """Test that config returns correct pipeline class."""
        config = ImageVectorSearchPipelineConfig(
            name="test_config",
            search_mode="multi",
        )

        assert config.get_pipeline_class() == ImageVectorSearchRetrievalPipeline

    def test_config_get_pipeline_kwargs_single(self):
        """Test that config returns correct pipeline kwargs for single mode."""
        config = ImageVectorSearchPipelineConfig(
            name="test_config",
            search_mode="single",
        )

        kwargs = config.get_pipeline_kwargs()

        assert kwargs["search_mode"] == "single"
        assert kwargs["embedding_model"] is None

    def test_config_get_pipeline_kwargs_multi(self):
        """Test that config returns correct pipeline kwargs for multi mode."""
        config = ImageVectorSearchPipelineConfig(
            name="test_config",
            search_mode="multi",
        )

        kwargs = config.get_pipeline_kwargs()

        assert kwargs["search_mode"] == "multi"
        assert kwargs["embedding_model"] is None

    def test_config_default_search_mode(self):
        """Test that config uses multi search mode by default."""
        config = ImageVectorSearchPipelineConfig(
            name="test_config",
        )

        assert config.search_mode == "multi"
