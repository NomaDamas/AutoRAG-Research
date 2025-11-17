"""Test cases for FileRepository.

Tests all FileRepository methods with atomic operations that leave no trace.
Uses existing database data for read operations and cleans up write operations.
"""

import pytest
from sqlalchemy.orm import Session

from autorag_research.orm.repository.file import FileRepository


@pytest.fixture
def file_repository(db_session: Session) -> FileRepository:
    """Create a FileRepository instance for testing.

    Args:
        db_session: Database session from conftest.py.

    Returns:
        FileRepository instance.
    """
    return FileRepository(db_session)


def test_get_by_path(file_repository: FileRepository, db_session: Session):
    """Test retrieving a file by its path."""
    # Use existing seed data (file id=1, path='/data/doc1.pdf')
    result = file_repository.get_by_path("/data/doc1.pdf")

    assert result is not None
    assert result.path == "/data/doc1.pdf"
    assert result.type == "raw"


def test_get_by_type(file_repository: FileRepository, db_session: Session):
    """Test retrieving files by type."""
    # Use existing seed data (files with type='raw' exist: ids 1-5)
    results = file_repository.get_by_type("raw")

    assert len(results) >= 5
    assert all(f.type == "raw" for f in results)


def test_get_with_documents(file_repository: FileRepository, db_session: Session):
    """Test retrieving a file with documents eagerly loaded."""
    # Use existing seed data (file id=1 has document id=1)
    result = file_repository.get_with_documents(1)

    assert result is not None
    assert result.id == 1
    assert hasattr(result, "documents")
    assert len(result.documents) >= 1


def test_get_with_pages(file_repository: FileRepository, db_session: Session):
    """Test retrieving a file with pages eagerly loaded."""
    # Use existing seed data (file id=6 is image with pages)
    result = file_repository.get_with_pages(6)

    assert result is not None
    assert result.id == 6
    assert hasattr(result, "pages")
    assert len(result.pages) >= 1


def test_get_with_image_chunks(file_repository: FileRepository, db_session: Session):
    """Test retrieving a file with image chunks eagerly loaded."""
    # Use existing seed data (file id=6 has image_chunks)
    result = file_repository.get_with_image_chunks(6)

    assert result is not None
    assert result.id == 6
    assert hasattr(result, "image_chunks")
    assert len(result.image_chunks) >= 1


def test_get_all_by_type(file_repository: FileRepository, db_session: Session):
    """Test retrieving all files by type with pagination."""
    # Use existing seed data (type='image' files exist: ids 6-10)
    results = file_repository.get_all_by_type("image", limit=10, offset=0)

    assert len(results) >= 5
    assert all(f.type == "image" for f in results)


def test_search_by_path_pattern(file_repository: FileRepository, db_session: Session):
    """Test searching files by path pattern."""
    # Use existing seed data (paths start with '/data/')
    results = file_repository.search_by_path_pattern("%/data/%")

    assert len(results) >= 10
    assert all("/data/" in f.path for f in results)


def test_count_by_type(file_repository: FileRepository, db_session: Session):
    """Test counting files by type."""
    # Use existing seed data (5 raw files exist)
    count = file_repository.count_by_type("raw")

    assert count >= 5


def test_get_all_types(file_repository: FileRepository, db_session: Session):
    """Test getting all unique file types."""
    # Use existing seed data (types: 'raw', 'image')
    types = file_repository.get_all_types()

    assert len(types) >= 2
    assert "raw" in types
    assert "image" in types
