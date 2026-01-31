from unittest.mock import patch

import pytest
from sqlalchemy.orm import sessionmaker, Session

from autorag_research.orm.repository.query import QueryRepository
from autorag_research.orm.schema import Chunk
from autorag_research.orm.service.retrieval_pipeline import RetrievalPipelineService


class TestRetrievalPipelineService:
    @pytest.fixture
    def service(self, session_factory):
        return RetrievalPipelineService(session_factory)

    @pytest.fixture
    def mock_retrieval_func(self):
        def retrieval_func(query_ids: list[int | str], top_k: int) -> list[list[dict]]:
            """Mock retrieval function that returns 2 results per query ID."""
            results = []
            for _ in query_ids:
                results.append(
                    [
                        {"doc_id": 1, "score": 0.9},
                        {"doc_id": 2, "score": 0.8},
                    ][:top_k]
                )
            return results

        return retrieval_func

    def test_run_pipeline(self, service, mock_retrieval_func, session_factory):
        # Count actual queries in database
        with session_factory() as session:
            query_repo = QueryRepository(session)
            query_count = query_repo.count()

        pipeline_id = service.save_pipeline(
            name="test_pipeline",
            config={"type": "test"},
        )

        result = service.run_pipeline(
            retrieval_func=mock_retrieval_func,
            pipeline_id=pipeline_id,
            top_k=2,
        )

        assert "pipeline_id" in result
        assert "total_queries" in result
        assert "total_results" in result
        assert result["pipeline_id"] == pipeline_id
        assert result["total_queries"] == query_count
        assert result["total_results"] == query_count * 2

        service.delete_pipeline_results(pipeline_id)

    def test_delete_pipeline_results(self, service, mock_retrieval_func, session_factory):
        # Count actual queries in database
        with session_factory() as session:
            query_repo = QueryRepository(session)
            query_count = query_repo.count()

        pipeline_id = service.save_pipeline(
            name="test_delete_pipeline",
            config={"type": "test"},
        )

        service.run_pipeline(
            retrieval_func=mock_retrieval_func,
            pipeline_id=pipeline_id,
            top_k=2,
        )

        # Use UoW to verify results before deletion
        with service._create_uow() as uow:
            results_before = uow.chunk_results.get_by_pipeline(pipeline_id)
            assert len(results_before) > 0

        deleted_count = service.delete_pipeline_results(pipeline_id)

        expected_results = query_count * 2
        assert deleted_count == expected_results

        # Use UoW to verify results after deletion
        with service._create_uow() as uow:
            results_after = uow.chunk_results.get_by_pipeline(pipeline_id)
            assert len(results_after) == 0


class TestVectorSearchByEmbedding:
    """Tests for RetrievalPipelineService.vector_search_by_embedding method."""

    @pytest.fixture
    def retrieval_service(self, session_factory: sessionmaker[Session]):
        """Create a RetrievalPipelineService instance for testing."""
        from autorag_research.orm.service.retrieval_pipeline import RetrievalPipelineService

        return RetrievalPipelineService(session_factory=session_factory)

    def test_vector_search_by_embedding(self, retrieval_service):
        """Test vector search using a provided embedding directly."""
        # Mock chunk results
        mock_chunk_results = [
            (Chunk(id=1, contents="Text content 1"), 0.1),  # distance
            (Chunk(id=2, contents="Text content 2"), 0.2),
        ]

        with patch("autorag_research.orm.repository.chunk.ChunkRepository.vector_search_with_scores") as mock_search:
            mock_search.return_value = mock_chunk_results

            test_embedding = [0.1] * 768
            results = retrieval_service.vector_search_by_embedding(embedding=test_embedding, top_k=3)

            assert len(results) == 2

            # Verify score conversion (1 - distance)
            assert results[0]["doc_id"] == 1
            assert results[0]["score"] == pytest.approx(0.9)  # 1 - 0.1
            assert results[0]["content"] == "Text content 1"

            assert results[1]["doc_id"] == 2
            assert results[1]["score"] == pytest.approx(0.8)  # 1 - 0.2

            # Verify vector_search_with_scores was called with correct embedding
            mock_search.assert_called_once()
            call_args = mock_search.call_args
            query_vector = call_args.kwargs.get("query_vector") or call_args.args[0]
            assert query_vector == test_embedding

    def test_vector_search_by_embedding_empty_results(self, retrieval_service):
        """Test vector search returns empty list when no results."""
        with patch("autorag_research.orm.repository.chunk.ChunkRepository.vector_search_with_scores") as mock_search:
            mock_search.return_value = []

            test_embedding = [0.1] * 768
            results = retrieval_service.vector_search_by_embedding(embedding=test_embedding, top_k=5)

            assert results == []
