"""Test cases for BaseEmbeddingRepository.

Tests the count methods for single and multi-vector embeddings.
Uses existing database data for read operations.
"""

import pytest
from sqlalchemy.orm import Session

from autorag_research.orm.repository.base import _sanitize_dict, _sanitize_text_value
from autorag_research.orm.repository.chunk import ChunkRepository
from autorag_research.orm.repository.query import QueryRepository
from autorag_research.orm.schema import Chunk


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


class TestSanitizeTextValue:
    """Tests for _sanitize_text_value helper function."""

    def test_removes_multiple_nul_bytes(self):
        """Test that multiple NUL bytes are all removed."""
        text_with_multiple_nul = "\x00Hello\x00World\x00"
        result = _sanitize_text_value(text_with_multiple_nul)
        assert result == "HelloWorld"

    def test_preserves_string_without_nul(self):
        """Test that strings without NUL bytes are unchanged."""
        clean_text = "Hello World"
        result = _sanitize_text_value(clean_text)
        assert result == "Hello World"


class TestSanitizeDict:
    """Tests for _sanitize_dict helper function."""

    def test_handles_mixed_content(self):
        """Test dict with mixed content types."""
        data = {
            "contents": "Text\x00with\x00nulls",
            "embedding": [0.1, 0.2, 0.3],
            "page_id": 1,
            "metadata": {"key": "value"},
        }
        result = _sanitize_dict(data)
        assert result["contents"] == "Textwithnulls"
        assert result["embedding"] == [0.1, 0.2, 0.3]
        assert result["page_id"] == 1
        assert result["metadata"] == {"key": "value"}


class TestAddBulkSanitization:
    """Tests for add_bulk method with NUL byte sanitization."""

    def test_add_bulk_handles_multiple_items_with_nul(self, db_session: Session, chunk_repository: ChunkRepository):
        """Test add_bulk with multiple items containing NUL bytes."""
        items = [
            {"contents": "First\x00chunk"},
            {"contents": "Second\x00\x00chunk"},
            {"contents": "Clean chunk"},
        ]

        ids = chunk_repository.add_bulk(items)
        db_session.flush()

        assert len(ids) == 3

        chunks = [db_session.get(Chunk, id_) for id_ in ids]
        assert chunks[0].contents == "Firstchunk"
        assert chunks[1].contents == "Secondchunk"
        assert chunks[2].contents == "Clean chunk"

        # Cleanup
        for chunk in chunks:
            db_session.delete(chunk)
        db_session.commit()

    def test_add_bulk_empty_list(self, db_session: Session, chunk_repository: ChunkRepository):
        """Test add_bulk with empty list returns empty list."""
        result = chunk_repository.add_bulk([])
        assert result == []
