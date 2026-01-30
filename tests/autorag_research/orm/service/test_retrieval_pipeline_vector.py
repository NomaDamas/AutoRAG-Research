"""Test cases for RetrievalPipelineService.vector_search method.

Tests the vector search functionality moved from VectorSearchModule to the service layer.
Supports both single-vector and multi-vector embeddings for text chunks.

Note: Some tests mock ChunkRepository methods due to a known issue where Vector objects
are double-wrapped when binding parameters.
"""

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.orm.schema import Chunk, Query
from autorag_research.orm.service.retrieval_pipeline import RetrievalPipelineService


@pytest.fixture
def retrieval_service(session_factory: sessionmaker[Session]) -> RetrievalPipelineService:
    """Create a RetrievalPipelineService instance for testing."""
    return RetrievalPipelineService(session_factory=session_factory)


class TestVectorSearchSingleVector:
    """Tests for single-vector search mode."""

    @pytest.fixture
    def test_query_with_embedding(self, db_session: Session):
        """Create a test query with embedding."""
        # Create test query with embedding (768-dim vector)
        base_embedding = [0.1] * 768
        query = Query(contents="What is machine learning?", embedding=base_embedding.copy())
        db_session.add(query)
        db_session.commit()

        yield query

        # Cleanup
        db_session.delete(query)
        db_session.commit()

    def test_single_vector_search(
        self,
        retrieval_service: RetrievalPipelineService,
        test_query_with_embedding: Query,
    ):
        """Test single-vector search using pre-computed query embedding."""
        # Mock chunk results (mocking repository due to known Vector double-wrap issue)
        mock_chunk_results = [
            (Chunk(id=1, contents="Text content 1"), 0.1),  # distance
            (Chunk(id=2, contents="Text content 2"), 0.2),
        ]

        with patch("autorag_research.orm.repository.chunk.ChunkRepository.vector_search_with_scores") as mock_search:
            mock_search.return_value = mock_chunk_results

            query_id = test_query_with_embedding.id
            results = retrieval_service.vector_search([query_id], top_k=3, search_mode="single")

            assert len(results) == 1
            assert len(results[0]) == 2

            # Verify score conversion (1 - distance)
            assert results[0][0]["doc_id"] == 1
            assert results[0][0]["score"] == pytest.approx(0.9)  # 1 - 0.1
            assert results[0][0]["content"] == "Text content 1"

            assert results[0][1]["doc_id"] == 2
            assert results[0][1]["score"] == pytest.approx(0.8)  # 1 - 0.2

            # Verify vector_search_with_scores was called with converted embedding
            mock_search.assert_called_once()
            call_args = mock_search.call_args
            query_vector = call_args.kwargs.get("query_vector") or call_args.args[0]
            assert isinstance(query_vector, list)
            assert len(query_vector) == 768
            # Check numeric type (could be float or numpy.float)
            assert all(isinstance(x, (int, float)) or hasattr(x, "__float__") for x in query_vector)

    def test_single_vector_raises_when_no_embedding(
        self,
        retrieval_service: RetrievalPipelineService,
        db_session: Session,
    ):
        """Test that single-vector search raises error when query has no embedding."""
        # Use seed data query which has no embedding
        query = db_session.get(Query, 1)
        assert query is not None
        assert query.embedding is None

        with pytest.raises(ValueError, match="has no embedding"):
            retrieval_service.vector_search([1], top_k=3, search_mode="single")


class TestVectorSearchMultiVector:
    """Tests for multi-vector (MaxSim) search mode."""

    @pytest.fixture
    def test_query_with_multi_embedding(self, db_session: Session):
        """Create a test query with multi-vector embeddings."""
        # Create test query with multi-vector embeddings
        query = Query(
            contents="What is machine learning?",
            embeddings=[[0.1] * 768, [0.12] * 768],  # 2 token vectors
        )
        db_session.add(query)
        db_session.commit()

        yield query

        # Cleanup
        db_session.delete(query)
        db_session.commit()

    def test_multi_vector_search(
        self,
        retrieval_service: RetrievalPipelineService,
        test_query_with_multi_embedding: Query,
    ):
        """Test multi-vector MaxSim search using pre-computed query embeddings."""
        # Mock chunk results from maxsim_search
        mock_chunk_results = [
            (Chunk(id=1, contents="Text content 1"), -0.8),  # MaxSim distance (lower = better)
            (Chunk(id=2, contents="Text content 2"), -0.6),
        ]

        with patch("autorag_research.orm.repository.chunk.ChunkRepository.maxsim_search") as mock_search:
            mock_search.return_value = mock_chunk_results

            query_id = test_query_with_multi_embedding.id
            results = retrieval_service.vector_search([query_id], top_k=2, search_mode="multi")

            assert len(results) == 1
            assert len(results[0]) == 2

            # Verify score conversion (-distance for MaxSim)
            assert results[0][0]["doc_id"] == 1
            assert results[0][0]["score"] == pytest.approx(0.8)  # -(-0.8)

            # Verify maxsim_search was called with converted embeddings
            mock_search.assert_called_once()
            call_args = mock_search.call_args
            query_vectors = call_args.kwargs.get("query_vectors") or call_args.args[0]
            assert isinstance(query_vectors, list)
            assert len(query_vectors) == 2
            assert all(isinstance(vec, list) for vec in query_vectors)
            assert all(isinstance(x, float) for vec in query_vectors for x in vec)

    def test_multi_vector_raises_when_no_embeddings(
        self,
        retrieval_service: RetrievalPipelineService,
        db_session: Session,
    ):
        """Test that multi-vector search raises error when query has no embeddings."""
        # Use seed data query which has no embeddings
        query = db_session.get(Query, 1)
        assert query is not None
        assert query.embeddings is None

        with pytest.raises(ValueError, match="has no multi-vector embeddings"):
            retrieval_service.vector_search([1], top_k=3, search_mode="multi")


class TestVectorSearchEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_query_ids(self, retrieval_service: RetrievalPipelineService):
        """Test vector_search with empty query ID list."""
        results = retrieval_service.vector_search([], top_k=5)
        assert results == []

    def test_multiple_query_ids(
        self,
        retrieval_service: RetrievalPipelineService,
        db_session: Session,
    ):
        """Test vector_search with multiple query IDs."""
        # Create multiple queries with embeddings
        base_embedding = [0.1] * 768
        queries = [Query(contents=f"Test query {i}", embedding=base_embedding.copy()) for i in range(3)]
        db_session.add_all(queries)
        db_session.commit()

        mock_chunk_results = [
            (Chunk(id=1, contents="Content"), 0.1),
        ]

        with patch("autorag_research.orm.repository.chunk.ChunkRepository.vector_search_with_scores") as mock_search:
            mock_search.return_value = mock_chunk_results

            query_ids = [q.id for q in queries]
            results = retrieval_service.vector_search(query_ids, top_k=3)

            assert len(results) == 3  # Three query IDs
            assert mock_search.call_count == 3

        # Cleanup
        for q in queries:
            db_session.delete(q)
        db_session.commit()

    def test_query_not_found(self, retrieval_service: RetrievalPipelineService):
        """Test that ValueError is raised when query ID is not found."""
        with pytest.raises(ValueError, match="Query 999999 not found"):
            retrieval_service.vector_search([999999], top_k=3)

    def test_string_query_ids(self, retrieval_service: RetrievalPipelineService):
        """Test vector_search with string query IDs (for string primary key schemas)."""
        # Mock query repository since we need string IDs which default schema doesn't support
        mock_query = MagicMock(spec=Query)
        mock_query.embedding = [0.1] * 768
        mock_query.embeddings = None

        with patch("autorag_research.orm.repository.query.QueryRepository.get_by_id") as mock_get:
            mock_get.return_value = mock_query

            with patch(
                "autorag_research.orm.repository.chunk.ChunkRepository.vector_search_with_scores"
            ) as mock_search:
                mock_search.return_value = [(Chunk(id=1, contents="Content"), 0.1)]
                results = retrieval_service.vector_search(["query-uuid-1"], top_k=3)

            assert len(results) == 1
            mock_get.assert_called_once_with("query-uuid-1")

    def test_default_search_mode_is_single(
        self,
        retrieval_service: RetrievalPipelineService,
        db_session: Session,
    ):
        """Test that default search_mode is 'single'."""
        # Create a query with embedding
        base_embedding = [0.1] * 768
        query = Query(contents="Test query", embedding=base_embedding.copy())
        db_session.add(query)
        db_session.commit()

        mock_chunk_results = [(Chunk(id=1, contents="Content"), 0.1)]

        try:
            with patch(
                "autorag_research.orm.repository.chunk.ChunkRepository.vector_search_with_scores"
            ) as mock_search:
                mock_search.return_value = mock_chunk_results

                # Call without specifying search_mode
                results = retrieval_service.vector_search([query.id], top_k=3)

                assert len(results) == 1
                # vector_search_with_scores should be called (not maxsim_search)
                mock_search.assert_called_once()
        finally:
            db_session.delete(query)
            db_session.commit()
