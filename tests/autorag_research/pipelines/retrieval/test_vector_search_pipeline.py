"""Test cases for VectorSearchRetrievalPipeline.

Tests the vector search retrieval pipeline logic using mocked vector search.
Supports single-vector (BaseEmbedding) and multi-vector (MultiVectorBaseEmbedding) modes.
"""

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.orm.repository.chunk_retrieved_result import ChunkRetrievedResultRepository
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

    @pytest.fixture
    def mock_single_vector_embedding(self):
        """Create a mock single-vector embedding model (BaseEmbedding)."""
        from llama_index.core.base.embeddings.base import BaseEmbedding

        mock = MagicMock(spec=BaseEmbedding)
        mock.model_name = "test-embedding-model"
        mock.get_text_embedding.return_value = [0.1, 0.2, 0.3]
        return mock

    @pytest.fixture
    def mock_multi_vector_embedding(self):
        """Create a mock multi-vector embedding model (MultiVectorBaseEmbedding)."""
        from autorag_research.embeddings.base import MultiVectorBaseEmbedding

        mock = MagicMock(spec=MultiVectorBaseEmbedding)
        mock.model_name = "test-multi-vector-model"
        mock.get_query_embedding.return_value = [[0.1, 0.2], [0.3, 0.4]]
        return mock

    def test_pipeline_creation_single_vector(
        self,
        session_factory: sessionmaker[Session],
        cleanup_pipeline_results: list[int],
        mock_single_vector_embedding: MagicMock,
    ):
        """Test that pipeline is created correctly with single-vector embedding."""
        from autorag_research.pipelines.retrieval.vector_search import (
            VectorSearchRetrievalPipeline,
        )

        pipeline = VectorSearchRetrievalPipeline(
            session_factory=session_factory,
            name="test_vector_search_single",
            embedding_model=mock_single_vector_embedding,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        assert pipeline.pipeline_id > 0
        assert pipeline.embedding_model == mock_single_vector_embedding

    def test_pipeline_creation_multi_vector(
        self,
        session_factory: sessionmaker[Session],
        cleanup_pipeline_results: list[int],
        mock_multi_vector_embedding: MagicMock,
    ):
        """Test that pipeline is created correctly with multi-vector embedding."""
        from autorag_research.pipelines.retrieval.vector_search import (
            VectorSearchRetrievalPipeline,
        )

        pipeline = VectorSearchRetrievalPipeline(
            session_factory=session_factory,
            name="test_vector_search_multi",
            embedding_model=mock_multi_vector_embedding,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        assert pipeline.pipeline_id > 0
        assert pipeline.embedding_model == mock_multi_vector_embedding

    def test_pipeline_config_single_vector(
        self,
        session_factory: sessionmaker[Session],
        cleanup_pipeline_results: list[int],
        mock_single_vector_embedding: MagicMock,
    ):
        """Test that pipeline config is correct for single-vector embedding."""
        from autorag_research.pipelines.retrieval.vector_search import (
            VectorSearchRetrievalPipeline,
        )

        pipeline = VectorSearchRetrievalPipeline(
            session_factory=session_factory,
            name="test_vector_search_config_single",
            embedding_model=mock_single_vector_embedding,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        config = pipeline._get_pipeline_config()
        assert config["type"] == "vector_search"
        assert config["embedding_model"] == "test-embedding-model"

    def test_pipeline_config_multi_vector(
        self,
        session_factory: sessionmaker[Session],
        cleanup_pipeline_results: list[int],
        mock_multi_vector_embedding: MagicMock,
    ):
        """Test that pipeline config is correct for multi-vector embedding."""
        from autorag_research.pipelines.retrieval.vector_search import (
            VectorSearchRetrievalPipeline,
        )

        pipeline = VectorSearchRetrievalPipeline(
            session_factory=session_factory,
            name="test_vector_search_config_multi",
            embedding_model=mock_multi_vector_embedding,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        config = pipeline._get_pipeline_config()
        assert config["type"] == "vector_search"
        assert config["embedding_model"] == "test-multi-vector-model"

    def test_pipeline_config_with_string_model_name(
        self,
        session_factory: sessionmaker[Session],
        cleanup_pipeline_results: list[int],
    ):
        """Test that pipeline config handles string model name correctly."""
        from autorag_research.pipelines.retrieval.vector_search import (
            VectorSearchRetrievalPipeline,
        )

        pipeline = VectorSearchRetrievalPipeline(
            session_factory=session_factory,
            name="test_vector_search_config_string",
            embedding_model="openai-large",  # String model name
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        config = pipeline._get_pipeline_config()
        assert config["embedding_model"] == "openai-large"

    def test_retrieve_single_query(
        self,
        session_factory: sessionmaker[Session],
        cleanup_pipeline_results: list[int],
        mock_single_vector_embedding: MagicMock,
    ):
        """Test single query retrieval with mocked vector search."""
        from autorag_research.pipelines.retrieval.vector_search import (
            VectorSearchRetrievalPipeline,
        )

        # Mock module results
        mock_module_results = [
            [
                {"doc_id": 1, "score": 0.95, "content": "Content 1"},
                {"doc_id": 2, "score": 0.85, "content": "Content 2"},
                {"doc_id": 3, "score": 0.75, "content": "Content 3"},
            ]
        ]

        with patch("autorag_research.nodes.retrieval.vector_search.VectorSearchModule.run") as mock_run:
            mock_run.return_value = mock_module_results

            pipeline = VectorSearchRetrievalPipeline(
                session_factory=session_factory,
                name="test_vector_search_retrieve",
                embedding_model=mock_single_vector_embedding,
            )
            cleanup_pipeline_results.append(pipeline.pipeline_id)

            results = pipeline.retrieve("test query", top_k=3)

            assert isinstance(results, list)
            assert len(results) == 3
            for i, result in enumerate(results):
                assert result["doc_id"] == mock_module_results[0][i]["doc_id"]
                assert result["score"] == mock_module_results[0][i]["score"]

    def test_run_full_pipeline(
        self,
        session_factory: sessionmaker[Session],
        cleanup_pipeline_results: list[int],
        mock_single_vector_embedding: MagicMock,
    ):
        """Test running the full pipeline with mocked vector search."""
        from autorag_research.pipelines.retrieval.vector_search import (
            VectorSearchRetrievalPipeline,
        )

        # Mock module results - return results for each of 5 seed queries
        mock_result = [
            {"doc_id": 1, "score": 0.9, "content": "Content 1"},
            {"doc_id": 2, "score": 0.8, "content": "Content 2"},
        ]
        mock_module_results = [mock_result for _ in range(5)]  # 5 queries in seed data

        with patch("autorag_research.nodes.retrieval.vector_search.VectorSearchModule.run") as mock_run:
            mock_run.return_value = mock_module_results

            pipeline = VectorSearchRetrievalPipeline(
                session_factory=session_factory,
                name="test_vector_search_full_run",
                embedding_model=mock_single_vector_embedding,
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

    def test_results_persisted_correctly(
        self,
        session_factory: sessionmaker[Session],
        cleanup_pipeline_results: list[int],
        mock_single_vector_embedding: MagicMock,
    ):
        """Test that results are correctly persisted in database."""
        from autorag_research.pipelines.retrieval.vector_search import (
            VectorSearchRetrievalPipeline,
        )

        # Mock module results
        mock_result = [
            {"doc_id": 1, "score": 0.95, "content": "Content 1"},
            {"doc_id": 2, "score": 0.85, "content": "Content 2"},
        ]
        mock_module_results = [mock_result for _ in range(5)]

        with patch("autorag_research.nodes.retrieval.vector_search.VectorSearchModule.run") as mock_run:
            mock_run.return_value = mock_module_results

            pipeline = VectorSearchRetrievalPipeline(
                session_factory=session_factory,
                name="test_vector_search_persistence",
                embedding_model=mock_single_vector_embedding,
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
                assert all(r.rel_score >= 0 for r in results)
            finally:
                session.close()


class TestVectorSearchPipelineConfig:
    """Tests for VectorSearchPipelineConfig dataclass."""

    def test_config_get_pipeline_class(self):
        """Test that config returns correct pipeline class."""
        from llama_index.core.base.embeddings.base import BaseEmbedding

        from autorag_research.pipelines.retrieval.vector_search import (
            VectorSearchPipelineConfig,
            VectorSearchRetrievalPipeline,
        )

        mock_embedding = MagicMock(spec=BaseEmbedding)

        config = VectorSearchPipelineConfig(
            name="test_config",
            embedding_model=mock_embedding,
        )

        assert config.get_pipeline_class() == VectorSearchRetrievalPipeline

    def test_config_get_pipeline_kwargs(self):
        """Test that config returns correct pipeline kwargs."""
        from llama_index.core.base.embeddings.base import BaseEmbedding

        from autorag_research.pipelines.retrieval.vector_search import (
            VectorSearchPipelineConfig,
        )

        mock_embedding = MagicMock(spec=BaseEmbedding)

        config = VectorSearchPipelineConfig(
            name="test_config",
            embedding_model=mock_embedding,
        )

        kwargs = config.get_pipeline_kwargs()

        assert kwargs["embedding_model"] == mock_embedding
