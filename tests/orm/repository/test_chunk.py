"""Test cases for ChunkRepository.

Tests all ChunkRepository methods with atomic operations that leave no trace.
Uses existing database data for read operations and cleans up write operations.
"""

import uuid

import pytest
from sqlalchemy.orm import Session

from autorag_research.orm.repository.chunk import ChunkRepository
from autorag_research.orm.schema import (
    Chunk,
)


@pytest.fixture
def chunk_repository(db_session: Session) -> ChunkRepository:
    """Create a ChunkRepository instance for testing.

    Args:
        db_session: Database session from conftest.py.

    Returns:
        ChunkRepository instance.
    """
    return ChunkRepository(db_session)


def test_get_by_caption_id(chunk_repository: ChunkRepository, db_session: Session):
    """Test retrieving all chunks for a specific caption."""
    # Use existing seed data (caption id=1 has chunks 1, 2)
    results = chunk_repository.get_by_caption_id(1)

    assert len(results) >= 2
    assert all(c.parent_caption == 1 for c in results)


def test_get_with_parent_caption(chunk_repository: ChunkRepository, db_session: Session):
    """Test retrieving a chunk with its parent caption eagerly loaded."""
    # Use existing seed data (chunk id=1, parent_caption=1)
    result = chunk_repository.get_with_parent_caption(1)

    assert result is not None
    assert result.id == 1
    assert result.parent_caption_obj is not None
    assert result.parent_caption_obj.id == 1


def test_get_with_caption_chunk_relations(chunk_repository: ChunkRepository, db_session: Session):
    """Test retrieving a chunk with its caption-chunk relations eagerly loaded."""
    # Use existing seed data (chunk id=1 has caption_chunk_relation)
    result = chunk_repository.get_with_caption_chunk_relations(1)

    assert result is not None
    assert result.id == 1
    assert hasattr(result, "caption_chunk_relations")
    assert len(result.caption_chunk_relations) >= 1


def test_get_with_retrieval_relations(chunk_repository: ChunkRepository, db_session: Session):
    """Test retrieving a chunk with its retrieval relations eagerly loaded."""
    # Use existing seed data (chunk id=1 has retrieval_relation with query id=1)
    result = chunk_repository.get_with_retrieval_relations(1)

    assert result is not None
    assert result.id == 1
    assert hasattr(result, "retrieval_relations")
    assert len(result.retrieval_relations) >= 1


def test_get_with_chunk_retrieved_results(chunk_repository: ChunkRepository, db_session: Session):
    """Test retrieving a chunk with its chunk retrieved results eagerly loaded."""
    # Use existing seed data (chunk id=1 has chunk_retrieved_result)
    result = chunk_repository.get_with_chunk_retrieved_results(1)

    assert result is not None
    assert result.id == 1
    assert hasattr(result, "chunk_retrieved_results")
    assert len(result.chunk_retrieved_results) >= 1


def test_search_by_contents(chunk_repository: ChunkRepository, db_session: Session):
    """Test searching chunks by contents using SQL LIKE."""
    # Use existing seed data (chunks contain "Chunk")
    results = chunk_repository.search_by_contents("Chunk")

    assert len(results) >= 6
    assert all("Chunk" in c.contents for c in results)


def test_get_by_contents_exact(chunk_repository: ChunkRepository, db_session: Session):
    """Test retrieving chunks with exact contents match."""
    # Use existing seed data (chunk id=1 has contents "Chunk 1-1")
    results = chunk_repository.get_by_contents_exact("Chunk 1-1")

    assert len(results) >= 1
    assert all(c.contents == "Chunk 1-1" for c in results)


def test_get_chunks_with_embeddings(chunk_repository: ChunkRepository, db_session: Session):
    """Test retrieving chunks that have embeddings."""
    # Create test data with embedding (seed data has NULL embeddings)
    unique_id = uuid.uuid4()
    embedding_vector = [0.1] * 768  # Convert to string for pgvector
    chunk = Chunk(parent_caption=None, contents=f"Chunk with embedding {unique_id}", embedding=embedding_vector)
    db_session.add(chunk)
    db_session.commit()

    # Test retrieval
    results = chunk_repository.get_chunks_with_embeddings(limit=10)

    assert len(results) >= 1
    assert len(results) <= 10
    assert all(c.embedding is not None for c in results)

    # Test with offset
    results_offset = chunk_repository.get_chunks_with_embeddings(limit=5, offset=1)
    assert len(results_offset) <= 5

    # Cleanup
    db_session.delete(chunk)
    db_session.commit()


def test_get_chunks_without_embeddings(chunk_repository: ChunkRepository, db_session: Session):
    """Test retrieving chunks that do not have embeddings."""
    # Use existing seed data (all chunks 1-6 have NULL embeddings)
    results = chunk_repository.get_chunks_without_embeddings(limit=10)

    assert len(results) <= 10
    assert all(c.embedding is None for c in results)

    # Test with offset
    results_offset = chunk_repository.get_chunks_without_embeddings(limit=5, offset=1)
    assert len(results_offset) <= 5


def test_count_by_caption(chunk_repository: ChunkRepository, db_session: Session):
    """Test counting the number of chunks for a specific caption."""
    # Use existing seed data (caption id=1 has 2 chunks)
    count = chunk_repository.count_by_caption(1)

    assert count >= 2


def test_get_with_all_relations(chunk_repository: ChunkRepository, db_session: Session):
    """Test retrieving a chunk with all relationships eagerly loaded."""
    # Use existing seed data (chunk id=1 has all relations)
    result = chunk_repository.get_with_all_relations(1)

    assert result is not None
    assert result.id == 1
    assert result.parent_caption_obj is not None
    assert hasattr(result, "caption_chunk_relations")
    assert hasattr(result, "retrieval_relations")
    assert hasattr(result, "chunk_retrieved_results")
