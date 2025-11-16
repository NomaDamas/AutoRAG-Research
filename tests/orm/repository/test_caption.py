"""Test cases for CaptionRepository.

Tests all CaptionRepository methods with atomic operations that leave no trace.
Uses existing database data for read operations and cleans up write operations.
"""

import pytest
from sqlalchemy.orm import Session

from autorag_research.orm.repository.caption import CaptionRepository


@pytest.fixture
def caption_repository(db_session: Session) -> CaptionRepository:
    """Create a CaptionRepository instance for testing.

    Args:
        db_session: Database session from conftest.py.

    Returns:
        CaptionRepository instance.
    """
    return CaptionRepository(db_session)


def test_get_by_page_id(caption_repository: CaptionRepository, db_session: Session):
    """Test retrieving all captions for a specific page."""
    # Use existing seed data (page id=1 has caption id=1)
    results = caption_repository.get_by_page_id(1)

    assert len(results) >= 1
    assert all(c.page_id == 1 for c in results)


def test_get_with_page(caption_repository: CaptionRepository, db_session: Session):
    """Test retrieving a caption with its page eagerly loaded."""
    # Use existing seed data (caption id=1, page_id=1)
    result = caption_repository.get_with_page(1)

    assert result is not None
    assert result.id == 1
    assert result.page is not None
    assert result.page.id == 1


def test_get_with_chunks(caption_repository: CaptionRepository, db_session: Session):
    """Test retrieving a caption with its chunks eagerly loaded."""
    # Use existing seed data (caption id=1 has chunks 1, 2)
    result = caption_repository.get_with_chunks(1)

    assert result is not None
    assert result.id == 1
    assert hasattr(result, "chunks")
    assert len(result.chunks) >= 2


def test_get_with_caption_chunk_relations(caption_repository: CaptionRepository, db_session: Session):
    """Test retrieving a caption with its caption-chunk relations eagerly loaded."""
    # Use existing seed data (caption id=1 has relations with chunks 1, 2)
    result = caption_repository.get_with_caption_chunk_relations(1)

    assert result is not None
    assert result.id == 1
    assert hasattr(result, "caption_chunk_relations")
    assert len(result.caption_chunk_relations) >= 2


def test_search_by_contents(caption_repository: CaptionRepository, db_session: Session):
    """Test searching captions by contents using SQL LIKE."""
    # Use existing seed data (captions contain "Caption for page")
    results = caption_repository.search_by_contents("Caption for page")

    assert len(results) >= 10
    assert all("Caption for page" in c.contents for c in results)


def test_get_all_with_page(caption_repository: CaptionRepository, db_session: Session):
    """Test retrieving all captions with their pages eagerly loaded."""
    # Use existing seed data (10 captions exist)
    results = caption_repository.get_all_with_page(limit=5)

    assert len(results) >= 1
    assert len(results) <= 5
    for result in results:
        assert hasattr(result, "page")
        assert result.page is not None

    # Test with offset
    results_offset = caption_repository.get_all_with_page(limit=5, offset=1)
    assert len(results_offset) <= 5


def test_count_by_page(caption_repository: CaptionRepository, db_session: Session):
    """Test counting the number of captions for a specific page."""
    # Use existing seed data (page id=1 has 1 caption)
    count = caption_repository.count_by_page(1)

    assert count >= 1


def test_get_by_contents_exact(caption_repository: CaptionRepository, db_session: Session):
    """Test retrieving captions with exact contents match."""
    # Use existing seed data (caption id=1 has specific content)
    results = caption_repository.get_by_contents_exact("Caption for page 1 of doc1")

    assert len(results) >= 1
    assert all(c.contents == "Caption for page 1 of doc1" for c in results)


def test_get_with_all_relations(caption_repository: CaptionRepository, db_session: Session):
    """Test retrieving a caption with all relationships eagerly loaded."""
    # Use existing seed data (caption id=1 has page, chunks, and relations)
    result = caption_repository.get_with_all_relations(1)

    assert result is not None
    assert result.id == 1
    assert result.page is not None
    assert hasattr(result, "chunks")
    assert hasattr(result, "caption_chunk_relations")
