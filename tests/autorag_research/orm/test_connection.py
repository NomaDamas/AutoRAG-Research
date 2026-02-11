"""Tests for autorag_research.orm.connection module."""

import copy

import pytest
from sqlalchemy import text

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


class TestDBConnectionCreateBm25Indexes:
    """Tests for BM25 index creation during create_schema."""

    def test_create_schema_creates_bm25_index(self, db_connection):
        """Test that create_schema creates the idx_chunk_bm25 index."""
        engine = db_connection.get_engine()

        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT indexname FROM pg_indexes WHERE indexname = 'idx_chunk_bm25'")
            ).fetchone()

        assert result is not None, "idx_chunk_bm25 index should exist after create_schema"


class TestDBConnectionRestoreDatabase:
    """Tests for DBConnection.restore_database method."""

    @pytest.mark.data
    def test_dump_and_restore(self, tmp_path):
        """Test that dump and restore preserves all data correctly."""
        conn = DBConnection.from_env()
        dump_file = tmp_path / "test_backup.dump"

        # Dump the database
        result = conn.dump_database(output_file=dump_file)

        assert result == dump_file
        assert dump_file.exists()
        assert dump_file.stat().st_size > 0

        # Restore the database from the dump
        restored_conn = copy.deepcopy(conn)
        restored_conn.database = "test_db_for_pg_restore"

        try:
            restored_conn.restore_database(dump_file, create=True)

            # Verify data integrity by comparing row counts and key data
            original_engine = conn.get_engine()
            restored_engine = restored_conn.get_engine()

            tables_to_verify = [
                ("file", 10),
                ("document", 5),
                ("page", 10),
                ("chunk", 8),
                ("query", 5),
                ("pipeline", 2),
                ("metric", 2),
            ]

            with original_engine.connect() as orig_conn, restored_engine.connect() as rest_conn:
                for table, expected_count in tables_to_verify:
                    # Compare row counts
                    orig_count = orig_conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()  # noqa: S608
                    rest_count = rest_conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()  # noqa: S608
                    assert orig_count == rest_count
                    assert rest_count == expected_count

                # Verify specific data: documents
                orig_docs = orig_conn.execute(
                    text("SELECT id, filename, author, title FROM document ORDER BY id")
                ).fetchall()
                rest_docs = rest_conn.execute(
                    text("SELECT id, filename, author, title FROM document ORDER BY id")
                ).fetchall()
                assert orig_docs == rest_docs, "Document data mismatch"

                # Verify specific data: queries
                orig_queries = orig_conn.execute(text("SELECT id, contents FROM query ORDER BY id")).fetchall()
                rest_queries = rest_conn.execute(text("SELECT id, contents FROM query ORDER BY id")).fetchall()
                assert orig_queries == rest_queries, "Query data mismatch"

                # Verify specific data: pipelines
                orig_pipelines = orig_conn.execute(text("SELECT id, name, config FROM pipeline ORDER BY id")).fetchall()
                rest_pipelines = rest_conn.execute(text("SELECT id, name, config FROM pipeline ORDER BY id")).fetchall()
                assert orig_pipelines == rest_pipelines, "Pipeline data mismatch"

                # Verify BM25 index exists after restore
                bm25_index = rest_conn.execute(
                    text("SELECT indexname FROM pg_indexes WHERE indexname = 'idx_chunk_bm25'")
                ).fetchone()
                assert bm25_index is not None, "idx_chunk_bm25 index should exist after restore"

                # Verify score column exists after restore (migration v1)
                score_col = rest_conn.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = 'retrieval_relation' AND column_name = 'score'"
                    )
                ).fetchone()
                assert score_col is not None, "score column should exist after restore"

        finally:
            # Cleanup: drop the restored database
            restored_conn.terminate_connections()
            restored_conn.drop_database()


class TestDBConnectionRunMigrations:
    """Tests for schema migration via DBConnection._run_migrations."""

    def test_retrieval_relation_score_column_exists(self, db_connection):
        """Test that the score column exists on retrieval_relation after migrations."""
        engine = db_connection.get_engine()

        with engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'retrieval_relation' AND column_name = 'score'"
                )
            ).fetchone()

        assert result is not None, "score column should exist on retrieval_relation"

    def test_run_migrations_is_idempotent(self, db_connection):
        """Test that calling _run_migrations multiple times is safe."""
        db_connection._run_migrations()
        db_connection._run_migrations()

        engine = db_connection.get_engine()

        with engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'retrieval_relation' AND column_name = 'score'"
                )
            ).fetchone()

        assert result is not None, "score column should exist after multiple migration runs"
