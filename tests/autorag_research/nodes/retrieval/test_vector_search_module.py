"""Test cases for VectorSearchModule.

Tests the vector search retrieval module supporting both single-vector
and multi-vector embeddings for text chunks using pre-computed query embeddings.
"""

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.orm.schema import Chunk, Query


class TestVectorSearchModuleSearchMode:
    """Tests for search mode selection."""

    def test_default_search_mode_is_single(self, session_factory: sessionmaker[Session]):
        """Test that default search_mode is 'single'."""
        from autorag_research.nodes.retrieval.vector_search import VectorSearchModule

        module = VectorSearchModule(session_factory=session_factory)

        assert module.search_mode == "single"

    def test_search_mode_single(self, session_factory: sessionmaker[Session]):
        """Test that search_mode can be set to 'single'."""
        from autorag_research.nodes.retrieval.vector_search import VectorSearchModule

        module = VectorSearchModule(
            session_factory=session_factory,
            search_mode="single",
        )

        assert module.search_mode == "single"

    def test_search_mode_multi(self, session_factory: sessionmaker[Session]):
        """Test that search_mode can be set to 'multi'."""
        from autorag_research.nodes.retrieval.vector_search import VectorSearchModule

        module = VectorSearchModule(
            session_factory=session_factory,
            search_mode="multi",
        )

        assert module.search_mode == "multi"


class TestVectorSearchModuleSingleVector:
    """Tests for single-vector search mode."""

    def test_single_vector_search(self, session_factory: sessionmaker[Session]):
        """Test single-vector search using pre-computed query embedding."""
        from autorag_research.nodes.retrieval.vector_search import VectorSearchModule

        # Mock query with pre-computed embedding
        mock_query = MagicMock(spec=Query)
        mock_query.embedding = [0.1, 0.2, 0.3]
        mock_query.embeddings = None

        # Mock chunk results
        mock_chunk_results = [
            (Chunk(id=1, contents="Text content 1"), 0.1),  # distance
            (Chunk(id=2, contents="Text content 2"), 0.2),
        ]

        module = VectorSearchModule(
            session_factory=session_factory,
            search_mode="single",
        )

        with (
            patch("autorag_research.orm.repository.query.QueryRepository.get_by_id") as mock_get,
            patch("autorag_research.orm.repository.chunk.ChunkRepository.vector_search_with_scores") as mock_search,
        ):
            mock_get.return_value = mock_query
            mock_search.return_value = mock_chunk_results

            results = module.run([1], top_k=3)

            assert len(results) == 1
            assert len(results[0]) == 2

            # Verify score conversion (1 - distance)
            assert results[0][0]["doc_id"] == 1
            assert results[0][0]["score"] == pytest.approx(0.9)  # 1 - 0.1
            assert results[0][0]["content"] == "Text content 1"

            assert results[0][1]["doc_id"] == 2
            assert results[0][1]["score"] == pytest.approx(0.8)  # 1 - 0.2

            # Verify query was fetched
            mock_get.assert_called_once_with(1)

    def test_single_vector_raises_when_no_embedding(self, session_factory: sessionmaker[Session]):
        """Test that single-vector search raises error when query has no embedding."""
        from autorag_research.nodes.retrieval.vector_search import VectorSearchModule

        # Mock query without embedding
        mock_query = MagicMock(spec=Query)
        mock_query.embedding = None
        mock_query.embeddings = None

        module = VectorSearchModule(
            session_factory=session_factory,
            search_mode="single",
        )

        with patch("autorag_research.orm.repository.query.QueryRepository.get_by_id") as mock_get:
            mock_get.return_value = mock_query

            with pytest.raises(ValueError, match="has no embedding"):
                module.run([1], top_k=3)


class TestVectorSearchModuleMultiVector:
    """Tests for multi-vector (MaxSim) search mode."""

    def test_multi_vector_search(self, session_factory: sessionmaker[Session]):
        """Test multi-vector MaxSim search using pre-computed query embeddings."""
        from autorag_research.nodes.retrieval.vector_search import VectorSearchModule

        # Mock query with pre-computed multi-vector embeddings
        mock_query = MagicMock(spec=Query)
        mock_query.embedding = None
        mock_query.embeddings = [[0.1, 0.2], [0.3, 0.4]]  # Multiple vectors

        # Mock chunk results from maxsim_search
        mock_chunk_results = [
            (Chunk(id=1, contents="Text content 1"), -0.8),  # MaxSim distance (lower = better)
            (Chunk(id=2, contents="Text content 2"), -0.6),
        ]

        module = VectorSearchModule(
            session_factory=session_factory,
            search_mode="multi",
        )

        with (
            patch("autorag_research.orm.repository.query.QueryRepository.get_by_id") as mock_get,
            patch("autorag_research.orm.repository.chunk.ChunkRepository.maxsim_search") as mock_search,
        ):
            mock_get.return_value = mock_query
            mock_search.return_value = mock_chunk_results

            results = module.run([1], top_k=3)

            assert len(results) == 1
            assert len(results[0]) == 2

            # Verify score conversion (-distance for MaxSim)
            assert results[0][0]["doc_id"] == 1
            assert results[0][0]["score"] == pytest.approx(0.8)  # -(-0.8)

            # Verify query was fetched
            mock_get.assert_called_once_with(1)

    def test_multi_vector_raises_when_no_embeddings(self, session_factory: sessionmaker[Session]):
        """Test that multi-vector search raises error when query has no embeddings."""
        from autorag_research.nodes.retrieval.vector_search import VectorSearchModule

        # Mock query without multi-vector embeddings
        mock_query = MagicMock(spec=Query)
        mock_query.embedding = [0.1, 0.2, 0.3]  # Has single embedding
        mock_query.embeddings = None  # But no multi-vector embeddings

        module = VectorSearchModule(
            session_factory=session_factory,
            search_mode="multi",
        )

        with patch("autorag_research.orm.repository.query.QueryRepository.get_by_id") as mock_get:
            mock_get.return_value = mock_query

            with pytest.raises(ValueError, match="has no multi-vector embeddings"):
                module.run([1], top_k=3)


class TestVectorSearchModuleEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_query_ids(self, session_factory: sessionmaker[Session]):
        """Test running with empty query ID list."""
        from autorag_research.nodes.retrieval.vector_search import VectorSearchModule

        module = VectorSearchModule(session_factory=session_factory)

        results = module.run([], top_k=5)
        assert results == []

    def test_multiple_query_ids(self, session_factory: sessionmaker[Session]):
        """Test running with multiple query IDs."""
        from autorag_research.nodes.retrieval.vector_search import VectorSearchModule

        # Mock queries with pre-computed embeddings
        mock_query = MagicMock(spec=Query)
        mock_query.embedding = [0.1, 0.2, 0.3]
        mock_query.embeddings = None

        mock_chunk_results = [
            (Chunk(id=1, contents="Content"), 0.1),
        ]

        module = VectorSearchModule(session_factory=session_factory)

        with (
            patch("autorag_research.orm.repository.query.QueryRepository.get_by_id") as mock_get,
            patch("autorag_research.orm.repository.chunk.ChunkRepository.vector_search_with_scores") as mock_search,
        ):
            mock_get.return_value = mock_query
            mock_search.return_value = mock_chunk_results

            results = module.run([1, 2, 3], top_k=3)

            assert len(results) == 3  # Three query IDs
            assert mock_get.call_count == 3

    def test_query_not_found(self, session_factory: sessionmaker[Session]):
        """Test that ValueError is raised when query ID is not found."""
        from autorag_research.nodes.retrieval.vector_search import VectorSearchModule

        module = VectorSearchModule(session_factory=session_factory)

        with patch("autorag_research.orm.repository.query.QueryRepository.get_by_id") as mock_get:
            mock_get.return_value = None

            with pytest.raises(ValueError, match="Query 999 not found"):
                module.run([999], top_k=3)

    def test_string_query_ids(self, session_factory: sessionmaker[Session]):
        """Test running with string query IDs (for string primary key schemas)."""
        from autorag_research.nodes.retrieval.vector_search import VectorSearchModule

        mock_query = MagicMock(spec=Query)
        mock_query.embedding = [0.1, 0.2, 0.3]
        mock_query.embeddings = None

        mock_chunk_results = [
            (Chunk(id=1, contents="Content"), 0.1),
        ]

        module = VectorSearchModule(session_factory=session_factory)

        with (
            patch("autorag_research.orm.repository.query.QueryRepository.get_by_id") as mock_get,
            patch("autorag_research.orm.repository.chunk.ChunkRepository.vector_search_with_scores") as mock_search,
        ):
            mock_get.return_value = mock_query
            mock_search.return_value = mock_chunk_results

            results = module.run(["query-uuid-1"], top_k=3)

            assert len(results) == 1
            mock_get.assert_called_once_with("query-uuid-1")
