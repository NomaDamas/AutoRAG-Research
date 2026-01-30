"""Test cases for VectorSearchModule.

Tests the vector search retrieval module supporting both single-vector
and multi-vector embeddings for text chunks using pre-computed query embeddings.

Note: Some tests mock ChunkRepository.vector_search_with_scores due to a known issue
in the repository method where Vector objects are double-wrapped when binding parameters.
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

    def test_single_vector_search(self, session_factory: sessionmaker[Session], test_query_with_embedding: Query):
        """Test single-vector search using pre-computed query embedding."""
        from autorag_research.nodes.retrieval.vector_search import VectorSearchModule

        # Mock chunk results (mocking repository due to known Vector double-wrap issue)
        mock_chunk_results = [
            (Chunk(id=1, contents="Text content 1"), 0.1),  # distance
            (Chunk(id=2, contents="Text content 2"), 0.2),
        ]

        module = VectorSearchModule(
            session_factory=session_factory,
            search_mode="single",
        )

        with patch("autorag_research.orm.repository.chunk.ChunkRepository.vector_search_with_scores") as mock_search:
            mock_search.return_value = mock_chunk_results

            query_id = test_query_with_embedding.id
            results = module.run([query_id], top_k=3)

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
            assert all(isinstance(x, float) for x in query_vector)

    def test_single_vector_raises_when_no_embedding(self, session_factory: sessionmaker[Session], db_session: Session):
        """Test that single-vector search raises error when query has no embedding."""
        from autorag_research.nodes.retrieval.vector_search import VectorSearchModule

        # Use seed data query which has no embedding
        query = db_session.get(Query, 1)
        assert query is not None
        assert query.embedding is None

        module = VectorSearchModule(session_factory=session_factory, search_mode="single")

        with pytest.raises(ValueError, match="has no embedding"):
            module.run([1], top_k=3)


class TestVectorSearchModuleMultiVector:
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

    def test_multi_vector_search(self, session_factory: sessionmaker[Session], test_query_with_multi_embedding: Query):
        """Test multi-vector MaxSim search using pre-computed query embeddings."""
        from autorag_research.nodes.retrieval.vector_search import VectorSearchModule

        # Mock chunk results from maxsim_search
        mock_chunk_results = [
            (Chunk(id=1, contents="Text content 1"), -0.8),  # MaxSim distance (lower = better)
            (Chunk(id=2, contents="Text content 2"), -0.6),
        ]

        module = VectorSearchModule(
            session_factory=session_factory,
            search_mode="multi",
        )

        with patch("autorag_research.orm.repository.chunk.ChunkRepository.maxsim_search") as mock_search:
            mock_search.return_value = mock_chunk_results

            query_id = test_query_with_multi_embedding.id
            results = module.run([query_id], top_k=2)

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

    def test_multi_vector_raises_when_no_embeddings(self, session_factory: sessionmaker[Session], db_session: Session):
        """Test that multi-vector search raises error when query has no embeddings."""
        from autorag_research.nodes.retrieval.vector_search import VectorSearchModule

        # Use seed data query which has no embeddings
        query = db_session.get(Query, 1)
        assert query is not None
        assert query.embeddings is None

        module = VectorSearchModule(session_factory=session_factory, search_mode="multi")

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

    def test_multiple_query_ids(self, session_factory: sessionmaker[Session], db_session: Session):
        """Test running with multiple query IDs."""
        from autorag_research.nodes.retrieval.vector_search import VectorSearchModule

        # Create multiple queries with embeddings
        base_embedding = [0.1] * 768
        queries = [Query(contents=f"Test query {i}", embedding=base_embedding.copy()) for i in range(3)]
        db_session.add_all(queries)
        db_session.commit()

        mock_chunk_results = [
            (Chunk(id=1, contents="Content"), 0.1),
        ]

        module = VectorSearchModule(session_factory=session_factory)

        with patch("autorag_research.orm.repository.chunk.ChunkRepository.vector_search_with_scores") as mock_search:
            mock_search.return_value = mock_chunk_results

            query_ids = [q.id for q in queries]
            results = module.run(query_ids, top_k=3)

            assert len(results) == 3  # Three query IDs
            assert mock_search.call_count == 3

        # Cleanup
        for q in queries:
            db_session.delete(q)
        db_session.commit()

    def test_query_not_found(self, session_factory: sessionmaker[Session]):
        """Test that ValueError is raised when query ID is not found."""
        from autorag_research.nodes.retrieval.vector_search import VectorSearchModule

        module = VectorSearchModule(session_factory=session_factory)

        with pytest.raises(ValueError, match="Query 999999 not found"):
            module.run([999999], top_k=3)

    def test_string_query_ids(self, session_factory: sessionmaker[Session]):
        """Test running with string query IDs (for string primary key schemas)."""
        from autorag_research.nodes.retrieval.vector_search import VectorSearchModule

        # Mock query repository since we need string IDs which default schema doesn't support
        mock_query = MagicMock(spec=Query)
        mock_query.embedding = [0.1] * 768
        mock_query.embeddings = None

        module = VectorSearchModule(session_factory=session_factory)

        with patch("autorag_research.orm.repository.query.QueryRepository.get_by_id") as mock_get:
            mock_get.return_value = mock_query

            with patch(
                "autorag_research.orm.repository.chunk.ChunkRepository.vector_search_with_scores"
            ) as mock_search:
                mock_search.return_value = [(Chunk(id=1, contents="Content"), 0.1)]
                results = module.run(["query-uuid-1"], top_k=3)

            assert len(results) == 1
            mock_get.assert_called_once_with("query-uuid-1")


class TestVectorSearchModuleSearch:
    """Tests for VectorSearchModule.search() method - direct embedding input."""

    def test_search_single_vector(self, session_factory: sessionmaker[Session]):
        """Test search with single-vector embeddings (auto-detected)."""
        from autorag_research.nodes.retrieval.vector_search import VectorSearchModule

        mock_chunk_results = [
            (Chunk(id=1, contents="Text content 1"), 0.1),
            (Chunk(id=2, contents="Text content 2"), 0.2),
        ]

        module = VectorSearchModule(session_factory=session_factory)

        with patch("autorag_research.orm.repository.chunk.ChunkRepository.vector_search_with_scores") as mock_search:
            mock_search.return_value = mock_chunk_results

            # Single-vector input: list of 768-dim vectors
            embedding = [0.1] * 768
            results = module.search([embedding], top_k=3)

            assert len(results) == 1
            assert len(results[0]) == 2

            # Verify score conversion (1 - distance)
            assert results[0][0]["doc_id"] == 1
            assert results[0][0]["score"] == pytest.approx(0.9)
            assert results[0][0]["content"] == "Text content 1"

            # Verify vector_search_with_scores was called
            mock_search.assert_called_once()
            call_args = mock_search.call_args
            query_vector = call_args.kwargs.get("query_vector") or call_args.args[0]
            assert isinstance(query_vector, list)
            assert len(query_vector) == 768

    def test_search_multi_vector(self, session_factory: sessionmaker[Session]):
        """Test search with multi-vector embeddings (auto-detected)."""
        from autorag_research.nodes.retrieval.vector_search import VectorSearchModule

        mock_chunk_results = [
            (Chunk(id=1, contents="Text content 1"), -0.8),
            (Chunk(id=2, contents="Text content 2"), -0.6),
        ]

        module = VectorSearchModule(session_factory=session_factory)

        with patch("autorag_research.orm.repository.chunk.ChunkRepository.maxsim_search") as mock_search:
            mock_search.return_value = mock_chunk_results

            # Multi-vector input: list of lists of 768-dim vectors
            token_embeddings = [[0.1] * 768, [0.12] * 768]  # 2 token vectors per query
            results = module.search([token_embeddings], top_k=2)

            assert len(results) == 1
            assert len(results[0]) == 2

            # Verify score conversion (-distance for MaxSim)
            assert results[0][0]["doc_id"] == 1
            assert results[0][0]["score"] == pytest.approx(0.8)

            # Verify maxsim_search was called
            mock_search.assert_called_once()
            call_args = mock_search.call_args
            query_vectors = call_args.kwargs.get("query_vectors") or call_args.args[0]
            assert isinstance(query_vectors, list)
            assert len(query_vectors) == 2

    def test_search_multiple_single_vectors(self, session_factory: sessionmaker[Session]):
        """Test search with multiple single-vector queries."""
        from autorag_research.nodes.retrieval.vector_search import VectorSearchModule

        mock_chunk_results = [
            (Chunk(id=1, contents="Content"), 0.1),
        ]

        module = VectorSearchModule(session_factory=session_factory)

        with patch("autorag_research.orm.repository.chunk.ChunkRepository.vector_search_with_scores") as mock_search:
            mock_search.return_value = mock_chunk_results

            # Multiple single-vector queries
            embeddings = [[0.1] * 768, [0.2] * 768, [0.3] * 768]
            results = module.search(embeddings, top_k=3)

            assert len(results) == 3
            assert mock_search.call_count == 3

    def test_search_multiple_multi_vectors(self, session_factory: sessionmaker[Session]):
        """Test search with multiple multi-vector queries."""
        from autorag_research.nodes.retrieval.vector_search import VectorSearchModule

        mock_chunk_results = [
            (Chunk(id=1, contents="Content"), -0.5),
        ]

        module = VectorSearchModule(session_factory=session_factory)

        with patch("autorag_research.orm.repository.chunk.ChunkRepository.maxsim_search") as mock_search:
            mock_search.return_value = mock_chunk_results

            # Multiple multi-vector queries
            multi_embeddings = [
                [[0.1] * 768, [0.12] * 768],
                [[0.2] * 768, [0.22] * 768],
            ]
            results = module.search(multi_embeddings, top_k=2)

            assert len(results) == 2
            assert mock_search.call_count == 2


class TestVectorSearchModuleSearchByText:
    """Tests for VectorSearchModule.search_by_text() method - text input with runtime embedding."""

    def test_search_by_text_empty_texts(self, session_factory: sessionmaker[Session]):
        """Test search_by_text with empty text list."""
        from llama_index.core.embeddings.mock_embed_model import MockEmbedding

        from autorag_research.nodes.retrieval.vector_search import VectorSearchModule

        module = VectorSearchModule(session_factory=session_factory)
        mock_embed_model = MockEmbedding(embed_dim=768)

        results = module.search_by_text([], embedding_model=mock_embed_model, top_k=5)
        assert results == []

    def test_search_by_text_single_query(self, session_factory: sessionmaker[Session]):
        """Test search_by_text with a single text query."""
        from llama_index.core.embeddings.mock_embed_model import MockEmbedding

        from autorag_research.nodes.retrieval.vector_search import VectorSearchModule

        mock_chunk_results = [
            (Chunk(id=1, contents="Text content 1"), 0.1),
            (Chunk(id=2, contents="Text content 2"), 0.2),
        ]

        module = VectorSearchModule(session_factory=session_factory)
        mock_embed_model = MockEmbedding(embed_dim=768)

        with patch("autorag_research.orm.repository.chunk.ChunkRepository.vector_search_with_scores") as mock_search:
            mock_search.return_value = mock_chunk_results

            results = module.search_by_text(["What is machine learning?"], embedding_model=mock_embed_model, top_k=3)

            assert len(results) == 1
            assert len(results[0]) == 2

            # Verify score conversion (1 - distance)
            assert results[0][0]["doc_id"] == 1
            assert results[0][0]["score"] == pytest.approx(0.9)

            # Verify search was called with embeddings
            mock_search.assert_called_once()
            call_args = mock_search.call_args
            query_vector = call_args.kwargs.get("query_vector") or call_args.args[0]
            assert isinstance(query_vector, list)
            assert len(query_vector) == 768

    def test_search_by_text_multiple_queries(self, session_factory: sessionmaker[Session]):
        """Test search_by_text with multiple text queries."""
        from llama_index.core.embeddings.mock_embed_model import MockEmbedding

        from autorag_research.nodes.retrieval.vector_search import VectorSearchModule

        mock_chunk_results = [
            (Chunk(id=1, contents="Content"), 0.1),
        ]

        module = VectorSearchModule(session_factory=session_factory)
        mock_embed_model = MockEmbedding(embed_dim=768)

        with patch("autorag_research.orm.repository.chunk.ChunkRepository.vector_search_with_scores") as mock_search:
            mock_search.return_value = mock_chunk_results

            texts = ["Query 1", "Query 2", "Query 3"]
            results = module.search_by_text(texts, embedding_model=mock_embed_model, top_k=3)

            assert len(results) == 3
            assert mock_search.call_count == 3

    def test_search_by_text_with_config_string(self, session_factory: sessionmaker[Session]):
        """Test search_by_text with config string (uses @with_embedding decorator)."""
        from autorag_research.nodes.retrieval.vector_search import VectorSearchModule

        mock_chunk_results = [
            (Chunk(id=1, contents="Text content"), 0.1),
        ]

        module = VectorSearchModule(session_factory=session_factory)

        with patch("autorag_research.orm.repository.chunk.ChunkRepository.vector_search_with_scores") as mock_search:
            mock_search.return_value = mock_chunk_results

            # Use "mock" config which loads MockEmbedding from config/embedding/mock.yaml
            results = module.search_by_text(["Test query"], embedding_model="mock", top_k=3)

            assert len(results) == 1
            mock_search.assert_called_once()
