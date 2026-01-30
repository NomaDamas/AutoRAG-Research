"""Test cases for VectorSearchModule.

Tests the vector search retrieval module supporting both single-vector
and multi-vector embeddings for text chunks.
"""

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.orm.schema import Chunk


class TestVectorSearchModuleModelDetection:
    """Tests for embedding model type detection."""

    def test_detects_single_vector_model(self, session_factory: sessionmaker[Session]):
        """Test that BaseEmbedding is detected as single-vector model."""
        from llama_index.core.base.embeddings.base import BaseEmbedding

        from autorag_research.nodes.retrieval.vector_search import VectorSearchModule

        # Create a mock BaseEmbedding
        mock_embedding = MagicMock(spec=BaseEmbedding)

        module = VectorSearchModule(
            session_factory=session_factory,
            embedding_model=mock_embedding,
        )

        assert module._is_multi_vector_model(mock_embedding) is False

    def test_detects_multi_vector_model(self, session_factory: sessionmaker[Session]):
        """Test that MultiVectorBaseEmbedding is detected as multi-vector model."""
        from autorag_research.embeddings.base import MultiVectorBaseEmbedding
        from autorag_research.nodes.retrieval.vector_search import VectorSearchModule

        # Create a mock MultiVectorBaseEmbedding
        mock_embedding = MagicMock(spec=MultiVectorBaseEmbedding)

        module = VectorSearchModule(
            session_factory=session_factory,
            embedding_model=mock_embedding,
        )

        assert module._is_multi_vector_model(mock_embedding) is True


class TestVectorSearchModuleSingleVector:
    """Tests for single-vector (BaseEmbedding) search."""

    def test_single_vector_text_search(self, session_factory: sessionmaker[Session]):
        """Test single-vector search for text chunks."""
        from llama_index.core.base.embeddings.base import BaseEmbedding

        from autorag_research.nodes.retrieval.vector_search import VectorSearchModule

        # Mock embedding model
        mock_embedding = MagicMock(spec=BaseEmbedding)
        mock_embedding.get_text_embedding.return_value = [0.1, 0.2, 0.3]

        # Mock chunk results
        mock_chunk_results = [
            (Chunk(id=1, contents="Text content 1"), 0.1),  # distance
            (Chunk(id=2, contents="Text content 2"), 0.2),
        ]

        module = VectorSearchModule(
            session_factory=session_factory,
            embedding_model=mock_embedding,
        )

        with patch("autorag_research.orm.repository.chunk.ChunkRepository.vector_search_with_scores") as mock_search:
            mock_search.return_value = mock_chunk_results

            results = module.run(["test query"], top_k=3, embedding_model=mock_embedding)

            assert len(results) == 1
            assert len(results[0]) == 2

            # Verify score conversion (1 - distance)
            assert results[0][0]["doc_id"] == 1
            assert results[0][0]["score"] == pytest.approx(0.9)  # 1 - 0.1
            assert results[0][0]["content"] == "Text content 1"

            assert results[0][1]["doc_id"] == 2
            assert results[0][1]["score"] == pytest.approx(0.8)  # 1 - 0.2

            # Verify embedding model was called
            mock_embedding.get_text_embedding.assert_called_once_with("test query")


class TestVectorSearchModuleMultiVector:
    """Tests for multi-vector (MultiVectorBaseEmbedding) search."""

    def test_multi_vector_text_search(self, session_factory: sessionmaker[Session]):
        """Test multi-vector MaxSim search for text chunks."""
        from autorag_research.embeddings.base import MultiVectorBaseEmbedding
        from autorag_research.nodes.retrieval.vector_search import VectorSearchModule

        # Mock multi-vector embedding model
        mock_embedding = MagicMock(spec=MultiVectorBaseEmbedding)
        mock_embedding.get_query_embedding.return_value = [[0.1, 0.2], [0.3, 0.4]]  # Multiple vectors

        # Mock chunk results from maxsim_search
        mock_chunk_results = [
            (Chunk(id=1, contents="Text content 1"), -0.8),  # MaxSim distance (lower = better)
            (Chunk(id=2, contents="Text content 2"), -0.6),
        ]

        module = VectorSearchModule(
            session_factory=session_factory,
            embedding_model=mock_embedding,
        )

        with patch("autorag_research.orm.repository.chunk.ChunkRepository.maxsim_search") as mock_search:
            mock_search.return_value = mock_chunk_results

            results = module.run(["test query"], top_k=3, embedding_model=mock_embedding)

            assert len(results) == 1
            assert len(results[0]) == 2

            # Verify score conversion (-distance for MaxSim)
            assert results[0][0]["doc_id"] == 1
            assert results[0][0]["score"] == pytest.approx(0.8)  # -(-0.8)

            # Verify embedding model was called with get_query_embedding
            mock_embedding.get_query_embedding.assert_called_once_with("test query")


class TestVectorSearchModuleEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_queries(self, session_factory: sessionmaker[Session]):
        """Test running with empty query list."""
        from llama_index.core.base.embeddings.base import BaseEmbedding

        from autorag_research.nodes.retrieval.vector_search import VectorSearchModule

        mock_embedding = MagicMock(spec=BaseEmbedding)

        module = VectorSearchModule(
            session_factory=session_factory,
            embedding_model=mock_embedding,
        )

        results = module.run([], top_k=5, embedding_model=mock_embedding)
        assert results == []

    def test_multiple_queries(self, session_factory: sessionmaker[Session]):
        """Test running with multiple queries."""
        from llama_index.core.base.embeddings.base import BaseEmbedding

        from autorag_research.nodes.retrieval.vector_search import VectorSearchModule

        mock_embedding = MagicMock(spec=BaseEmbedding)
        mock_embedding.get_text_embedding.return_value = [0.1, 0.2, 0.3]

        mock_chunk_results = [
            (Chunk(id=1, contents="Content"), 0.1),
        ]

        module = VectorSearchModule(
            session_factory=session_factory,
            embedding_model=mock_embedding,
        )

        with patch("autorag_research.orm.repository.chunk.ChunkRepository.vector_search_with_scores") as mock_search:
            mock_search.return_value = mock_chunk_results

            results = module.run(["query1", "query2", "query3"], top_k=3, embedding_model=mock_embedding)

            assert len(results) == 3  # Three queries
            assert mock_embedding.get_text_embedding.call_count == 3

