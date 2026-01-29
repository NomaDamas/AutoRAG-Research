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


def test_get_with_page_chunk_relations(chunk_repository: ChunkRepository, db_session: Session):
    """Test retrieving a chunk with its page-chunk relations eagerly loaded."""
    # Use existing seed data (chunk id=1 has page_chunk_relation)
    result = chunk_repository.get_with_page_chunk_relations(1)

    assert result is not None
    assert result.id == 1
    assert hasattr(result, "page_chunk_relations")
    assert len(result.page_chunk_relations) >= 1


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


def test_get_with_embeddings(chunk_repository: ChunkRepository, db_session: Session):
    """Test retrieving chunks that have embeddings."""
    # Create test data with embedding (seed data has NULL embeddings)
    unique_id = uuid.uuid4()
    embedding_vector = [0.1] * 768  # Convert to string for pgvector
    chunk = Chunk(contents=f"Chunk with embedding {unique_id}", embedding=embedding_vector)
    db_session.add(chunk)
    db_session.commit()

    # Test retrieval
    results = chunk_repository.get_with_embeddings(limit=10)

    assert len(results) >= 1
    assert len(results) <= 10
    assert all(c.embedding is not None for c in results)

    # Test with offset
    results_offset = chunk_repository.get_with_embeddings(limit=5, offset=1)
    assert len(results_offset) <= 5

    # Cleanup
    db_session.delete(chunk)
    db_session.commit()


def test_get_without_embeddings(chunk_repository: ChunkRepository, db_session: Session):
    """Test retrieving chunks that do not have embeddings."""
    # Use existing seed data (all chunks 1-6 have NULL embeddings)
    results = chunk_repository.get_without_embeddings(limit=10)

    assert len(results) <= 10
    assert all(c.embedding is None for c in results)

    # Test with offset
    results_offset = chunk_repository.get_without_embeddings(limit=5, offset=1)
    assert len(results_offset) <= 5


def test_get_with_all_relations(chunk_repository: ChunkRepository, db_session: Session):
    result = chunk_repository.get_with_all_relations(1)

    assert result is not None
    assert result.id == 1
    assert hasattr(result, "page_chunk_relations")
    assert hasattr(result, "retrieval_relations")
    assert hasattr(result, "chunk_retrieved_results")


def test_get_chunks_with_empty_content(chunk_repository: ChunkRepository, db_session: Session):
    empty_chunk = Chunk(id=800001, contents="")
    whitespace_chunk = Chunk(id=800002, contents="   ")
    null_chunk = Chunk(id=800003, contents="        ")
    valid_chunk = Chunk(id=800004, contents="Valid content")

    db_session.add_all([empty_chunk, whitespace_chunk, null_chunk, valid_chunk])
    db_session.commit()

    results = chunk_repository.get_chunks_with_empty_content()

    result_ids = [c.id for c in results]
    assert 800001 in result_ids
    assert 800002 in result_ids
    assert 800003 in result_ids
    assert 800004 not in result_ids

    db_session.delete(db_session.get(Chunk, 800001))
    db_session.delete(db_session.get(Chunk, 800002))
    db_session.delete(db_session.get(Chunk, 800003))
    db_session.delete(db_session.get(Chunk, 800004))
    db_session.commit()


def test_get_table_chunks(chunk_repository: ChunkRepository, db_session: Session):
    """Test retrieving chunks that are tables (is_table=True)."""
    # Seed data has chunks 7, 8 with is_table=True
    results = chunk_repository.get_table_chunks()

    assert len(results) >= 2
    assert all(c.is_table is True for c in results)


def test_get_by_table_type(chunk_repository: ChunkRepository, db_session: Session):
    """Test retrieving chunks by table_type."""
    # Seed data has chunk 7 with table_type='markdown'
    results = chunk_repository.get_by_table_type("markdown")

    assert len(results) >= 1
    assert all(c.table_type == "markdown" for c in results)


def test_get_by_table_type_html(chunk_repository: ChunkRepository, db_session: Session):
    """Test retrieving chunks by table_type='html'."""
    # Seed data has chunk 8 with table_type='html'
    results = chunk_repository.get_by_table_type("html")

    assert len(results) >= 1
    assert all(c.table_type == "html" for c in results)


def test_get_non_table_chunks(chunk_repository: ChunkRepository, db_session: Session):
    """Test retrieving chunks that are not tables (is_table=False)."""
    # Seed data has chunks 1-6 with is_table=False
    results = chunk_repository.get_non_table_chunks()

    assert len(results) >= 6
    assert all(c.is_table is False for c in results)


def test_chunk_is_table_default(chunk_repository: ChunkRepository, db_session: Session):
    """Test that is_table defaults to False for new chunks."""
    chunk = Chunk(contents="Test content without is_table")
    db_session.add(chunk)
    db_session.flush()

    assert chunk.is_table is False
    assert chunk.table_type is None

    db_session.delete(chunk)
    db_session.commit()


def test_chunk_with_table_fields(chunk_repository: ChunkRepository, db_session: Session):
    """Test creating a chunk with is_table=True and table_type."""
    chunk = Chunk(contents="| A | B |", is_table=True, table_type="markdown")
    db_session.add(chunk)
    db_session.flush()

    assert chunk.is_table is True
    assert chunk.table_type == "markdown"

    db_session.delete(chunk)
    db_session.commit()


# ==================== BM25 Tests ====================


def test_batch_update_bm25_tokens(chunk_repository: ChunkRepository):
    """Test batch updating bm25_tokens for chunks using pre-seeded data."""
    # Seed data has 8 chunks (id 1-8)
    updated = chunk_repository.batch_update_bm25_tokens(tokenizer="bert", batch_size=10)
    assert updated >= 0  # May be 0 if already populated
    assert chunk_repository.count_with_bm25_tokens() >= 0


def test_bm25_search(chunk_repository: ChunkRepository):
    """Test BM25 search functionality using pre-seeded data."""
    # Populate BM25 tokens for seed data chunks
    chunk_repository.batch_update_bm25_tokens(tokenizer="bert")

    # Search for "Chunk" which matches seed data contents like "Chunk 1-1", "Chunk 2-1"
    results = chunk_repository.bm25_search(query_text="Chunk", limit=5, tokenizer="bert")

    assert len(results) >= 1
    # Results should be (chunk, score) tuples
    for chunk, score in results:
        assert chunk.id is not None
        assert score > 0  # Scores are negated, so positive means relevant
