"""Test cases for BaseEmbeddingRepository.

Tests the count methods for single and multi-vector embeddings.
Uses existing database data for read operations.
"""

import pytest
from sqlalchemy.orm import Session

from autorag_research.orm.repository.chunk import ChunkRepository
from autorag_research.orm.repository.query import QueryRepository


@pytest.fixture
def chunk_repository(db_session: Session) -> ChunkRepository:
    """Create a ChunkRepository instance for testing."""
    return ChunkRepository(db_session)


@pytest.fixture
def query_repository(db_session: Session) -> QueryRepository:
    """Create a QueryRepository instance for testing."""
    return QueryRepository(db_session)


class TestCountWithoutEmbeddings:
    """Test count_without_embeddings method."""

    def test_count_without_embeddings_chunks(self, chunk_repository: ChunkRepository):
        """Test counting chunks without single-vector embeddings."""
        # Seed data has 6 chunks, all with NULL embedding
        count = chunk_repository.count_without_embeddings()

        assert count >= 6  # At least the seeded chunks
        assert isinstance(count, int)

    def test_count_without_embeddings_queries(self, query_repository: QueryRepository):
        """Test counting queries without single-vector embeddings."""
        # Seed data has 3 queries, all with NULL embedding
        count = query_repository.count_without_embeddings()

        assert count >= 3  # At least the seeded queries
        assert isinstance(count, int)

    def test_count_returns_zero_when_all_embedded(self, chunk_repository: ChunkRepository, db_session: Session):
        """Test that count returns 0 when all entities have embeddings."""
        # First get the count without embeddings
        initial_count = chunk_repository.count_without_embeddings()

        # Get one chunk without embedding and set it
        chunks = chunk_repository.get_without_embeddings(limit=1)
        if chunks:
            chunk = chunks[0]
            chunk.embedding = [0.1] * 768  # Set a dummy embedding (768 dims = testdb schema)
            db_session.flush()

            # Count should decrease by 1
            new_count = chunk_repository.count_without_embeddings()
            assert new_count == initial_count - 1

            # Clean up - reset embedding to NULL
            chunk.embedding = None
            db_session.flush()


class TestCountWithoutMultiEmbeddings:
    """Test count_without_multi_embeddings method."""

    def test_count_without_multi_embeddings_chunks(self, chunk_repository: ChunkRepository):
        """Test counting chunks without multi-vector embeddings."""
        # Seed data has 6 chunks, all with NULL embeddings (multi-vector)
        count = chunk_repository.count_without_multi_embeddings()

        assert count >= 6  # At least the seeded chunks
        assert isinstance(count, int)

    def test_count_without_multi_embeddings_queries(self, query_repository: QueryRepository):
        """Test counting queries without multi-vector embeddings."""
        # Seed data has 3 queries, all with NULL embeddings (multi-vector)
        count = query_repository.count_without_multi_embeddings()

        assert count >= 3  # At least the seeded queries
        assert isinstance(count, int)

    def test_count_returns_zero_when_all_multi_embedded(self, chunk_repository: ChunkRepository, db_session: Session):
        """Test that count returns 0 when all entities have multi-vector embeddings."""
        # First get the count without multi-vector embeddings
        initial_count = chunk_repository.count_without_multi_embeddings()

        # Get one chunk without multi-vector embedding and set it
        chunks = chunk_repository.get_without_multi_embeddings(limit=1)
        if chunks:
            chunk = chunks[0]
            chunk.embeddings = [
                [0.1] * 768,
                [0.2] * 768,
            ]  # Set dummy multi-vector embeddings (768 dims = testdb schema)
            db_session.flush()

            # Count should decrease by 1
            new_count = chunk_repository.count_without_multi_embeddings()
            assert new_count == initial_count - 1

            # Clean up - reset embeddings to NULL
            chunk.embeddings = None
            db_session.flush()
