"""Tests for autorag_research.orm.connection module."""

import pytest

from autorag_research.orm.connection import DBConnection


class TestDBConnectionDumpDatabase:
    """Tests for DBConnection.dump_database method."""

    @pytest.mark.data
    def test_dump_database_creates_file(self, tmp_path):
        """Test that dump_database creates a dump file from the test database."""
        conn = DBConnection.from_env()
        output_file = tmp_path / "test_backup.dump"

        result = conn.dump_database(output_file=output_file)

        assert result == output_file
        assert output_file.exists()
        assert output_file.stat().st_size > 0
