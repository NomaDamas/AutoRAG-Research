"""Test cases for DocumentRepository.

Tests all DocumentRepository methods with atomic operations that leave no trace.
Uses existing database data for read operations and cleans up write operations.
"""

import pytest
from sqlalchemy.orm import Session

from autorag_research.orm.repository.document import DocumentRepository


@pytest.fixture
def document_repository(db_session: Session) -> DocumentRepository:
    return DocumentRepository(db_session)


def test_get_by_filename(document_repository: DocumentRepository):
    """Test retrieving a document by its filename."""
    # Use existing seed data (document id=1, filename='doc1.pdf')
    result = document_repository.get_by_filename("doc1.pdf")

    assert result is not None
    assert result.filename == "doc1.pdf"
    assert result.author == "alice"
    assert result.title == "Doc One"


def test_get_by_title(document_repository: DocumentRepository):
    """Test retrieving a document by its title."""
    # Use existing seed data (document id=2, title='Doc Two')
    result = document_repository.get_by_title("Doc Two")

    assert result is not None
    assert result.title == "Doc Two"
    assert result.author == "bob"
    assert result.filename == "doc2.pdf"


def test_get_by_author(document_repository: DocumentRepository):
    """Test retrieving all documents by a specific author."""
    # Use existing seed data (author='carol' has document id=3)
    results = document_repository.get_by_author("carol")

    assert len(results) >= 1
    assert all(doc.author == "carol" for doc in results)
    assert any(doc.filename == "doc3.pdf" for doc in results)


def test_get_with_pages(document_repository: DocumentRepository):
    """Test retrieving a document with its pages eagerly loaded."""
    # Use existing seed data (document id=1 has 2 pages)
    result = document_repository.get_with_pages(1)

    assert result is not None
    assert result.id == 1
    assert hasattr(result, "pages")
    assert len(result.pages) >= 2


def test_get_with_file(document_repository: DocumentRepository):
    """Test retrieving a document with its file eagerly loaded."""
    # Use existing seed data (document id=1, filepath=1)
    result = document_repository.get_with_file(1)

    assert result is not None
    assert result.id == 1
    assert hasattr(result, "file")
    assert result.file is not None
    assert result.file.id == 1


def test_get_all_with_pages(document_repository: DocumentRepository):
    """Test retrieving all documents with their pages eagerly loaded."""
    # Use existing seed data (5 documents exist, each with 2 pages)
    results = document_repository.get_all_with_pages(limit=10, offset=0)

    assert len(results) >= 5
    for doc in results:
        assert hasattr(doc, "pages")
        assert len(doc.pages) >= 2


def test_search_by_metadata(document_repository: DocumentRepository):
    """Test searching documents by metadata field."""
    # Use existing seed data (document id=1, doc_metadata={"topic": "alpha"})
    results = document_repository.search_by_metadata("topic", "alpha")

    assert len(results) >= 1
    assert any(doc.filename == "doc1.pdf" for doc in results)
    assert all(doc.doc_metadata.get("topic") == "alpha" for doc in results if doc.doc_metadata)


def test_count_pages(document_repository: DocumentRepository):
    """Test counting the number of pages in a document."""
    # Use existing seed data (document id=1 has 2 pages)
    count = document_repository.count_pages(1)

    assert count == 2
