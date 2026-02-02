"""Test cases for base repository module.

Tests sanitization functions that handle PostgreSQL NUL byte restrictions.
"""

from sqlalchemy.orm import Session

from autorag_research.orm.repository.base import (
    GenericRepository,
    _sanitize_dict,
    _sanitize_text_value,
)
from autorag_research.orm.schema import Chunk


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

    def test_add_bulk_handles_multiple_items_with_nul(self, db_session: Session):
        """Test add_bulk with multiple items containing NUL bytes."""
        repo = GenericRepository(db_session, Chunk)

        items = [
            {"contents": "First\x00chunk"},
            {"contents": "Second\x00\x00chunk"},
            {"contents": "Clean chunk"},
        ]

        ids = repo.add_bulk(items)
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

    def test_add_bulk_empty_list(self, db_session: Session):
        """Test add_bulk with empty list returns empty list."""
        repo = GenericRepository(db_session, Chunk)
        result = repo.add_bulk([])
        assert result == []
