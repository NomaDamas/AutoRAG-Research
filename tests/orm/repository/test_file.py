"""Test cases for FileRepository.

Tests all FileRepository methods with atomic operations that leave no trace.
Uses existing database data for read operations and cleans up write operations.
"""

import uuid

import pytest
from sqlalchemy.orm import Session

from autorag_research.orm.repository.file import FileRepository
from autorag_research.orm.schema import File


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
    # Create a file
    unique_path = f"/test/path/file_{uuid.uuid4()}.txt"
    file = File(type="raw", path=unique_path)
    db_session.add(file)
    db_session.commit()

    # Test retrieval
    result = file_repository.get_by_path(unique_path)

    assert result is not None
    assert result.path == unique_path

    # Cleanup
    db_session.delete(file)
    db_session.commit()


def test_get_by_type(file_repository: FileRepository, db_session: Session):
    """Test retrieving files by type."""
    # Create files of specific type
    unique_id = uuid.uuid4()
    files = [
        File(type="audio", path=f"/test/path/audio1_{unique_id}.mp3"),
        File(type="audio", path=f"/test/path/audio2_{unique_id}.mp3"),
    ]
    db_session.add_all(files)
    db_session.commit()

    # Test retrieval
    results = file_repository.get_by_type("audio")

    assert len(results) >= 2
    assert all(f.type == "audio" for f in results)

    # Cleanup
    for f in files:
        db_session.delete(f)
    db_session.commit()


def test_get_with_documents(file_repository: FileRepository, db_session: Session):
    """Test retrieving a file with documents eagerly loaded."""
    # Create a file
    unique_path = f"/test/path/file_{uuid.uuid4()}.txt"
    file = File(type="raw", path=unique_path)
    db_session.add(file)
    db_session.commit()
    file_id = file.id

    # Test retrieval
    result = file_repository.get_with_documents(file_id)

    assert result is not None
    assert result.id == file_id
    # Documents relationship is loaded (even if empty)
    assert hasattr(result, "documents")

    # Cleanup
    db_session.delete(file)
    db_session.commit()


def test_get_with_pages(file_repository: FileRepository, db_session: Session):
    """Test retrieving a file with pages eagerly loaded."""
    # Create a file
    unique_path = f"/test/path/file_{uuid.uuid4()}.txt"
    file = File(type="image", path=unique_path)
    db_session.add(file)
    db_session.commit()
    file_id = file.id

    # Test retrieval
    result = file_repository.get_with_pages(file_id)

    assert result is not None
    assert result.id == file_id
    # Pages relationship is loaded (even if empty)
    assert hasattr(result, "pages")

    # Cleanup
    db_session.delete(file)
    db_session.commit()


def test_get_with_image_chunks(file_repository: FileRepository, db_session: Session):
    """Test retrieving a file with image chunks eagerly loaded."""
    # Create a file
    unique_path = f"/test/path/file_{uuid.uuid4()}.png"
    file = File(type="image", path=unique_path)
    db_session.add(file)
    db_session.commit()
    file_id = file.id

    # Test retrieval
    result = file_repository.get_with_image_chunks(file_id)

    assert result is not None
    assert result.id == file_id
    # Image chunks relationship is loaded (even if empty)
    assert hasattr(result, "image_chunks")

    # Cleanup
    db_session.delete(file)
    db_session.commit()


def test_get_all_by_type(file_repository: FileRepository, db_session: Session):
    """Test retrieving all files by type with pagination."""
    # Create files
    unique_id = uuid.uuid4()
    files = [
        File(type="video", path=f"/test/path/video1_{unique_id}.mp4"),
        File(type="video", path=f"/test/path/video2_{unique_id}.mp4"),
    ]
    db_session.add_all(files)
    db_session.commit()

    # Test retrieval with pagination
    results = file_repository.get_all_by_type("video", limit=10, offset=0)

    assert len(results) >= 2
    assert all(f.type == "video" for f in results)

    # Cleanup
    for f in files:
        db_session.delete(f)
    db_session.commit()


def test_search_by_path_pattern(file_repository: FileRepository, db_session: Session):
    """Test searching files by path pattern."""
    # Create files with specific pattern
    unique_id = uuid.uuid4()
    search_pattern = f"test_pattern_{unique_id}"
    files = [
        File(type="raw", path=f"/test/{search_pattern}/file1.txt"),
        File(type="raw", path=f"/test/{search_pattern}/file2.txt"),
    ]
    db_session.add_all(files)
    db_session.commit()

    # Test search
    results = file_repository.search_by_path_pattern(f"%{search_pattern}%")

    assert len(results) >= 2
    assert all(search_pattern in f.path for f in results)

    # Cleanup
    for f in files:
        db_session.delete(f)
    db_session.commit()


def test_count_by_type(file_repository: FileRepository, db_session: Session):
    """Test counting files by type."""
    # Create files of specific type
    unique_id = uuid.uuid4()
    files = [
        File(type="raw", path=f"/test/path/count1_{unique_id}.txt"),
        File(type="raw", path=f"/test/path/count2_{unique_id}.txt"),
    ]
    db_session.add_all(files)
    db_session.commit()

    # Test count
    count = file_repository.count_by_type("raw")

    assert count >= 2

    # Cleanup
    for f in files:
        db_session.delete(f)
    db_session.commit()


def test_get_all_types(file_repository: FileRepository, db_session: Session):
    """Test getting all unique file types."""
    # Create files with different types
    unique_id = uuid.uuid4()
    files = [
        File(type="raw", path=f"/test/path/types1_{unique_id}.txt"),
        File(type="image", path=f"/test/path/types2_{unique_id}.png"),
    ]
    db_session.add_all(files)
    db_session.commit()

    # Test retrieval
    types = file_repository.get_all_types()

    assert len(types) > 0
    assert "raw" in types or "image" in types

    # Cleanup
    for f in files:
        db_session.delete(f)
    db_session.commit()
