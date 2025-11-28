"""Test cases for restore_database function.

Tests database dump and restore functionality using Docker PostgreSQL.
Creates temporary databases, dumps them, restores to new databases,
and verifies data integrity.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import psycopg
import pytest

from autorag_research.data.restore import restore_database
from autorag_research.orm.util import create_database, drop_database

# Database connection parameters from environment or defaults
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = int(os.getenv("PG_PORT", "5432"))
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")

# Test database names
SOURCE_DB = "test_restore_source"
TARGET_DB = "test_restore_target"


@pytest.fixture
def source_database():
    """Create a source database with sample schema and data.

    Yields the database name after setup.
    Drops the database after the test completes.
    """
    # Create the source database
    create_database(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=SOURCE_DB,
    )

    # Connect and create sample schema + data
    with psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        dbname=SOURCE_DB,
    ) as conn:
        with conn.cursor() as cursor:
            # Install vector extension
            cursor.execute("""
                DO $$
                BEGIN
                    BEGIN
                        CREATE EXTENSION IF NOT EXISTS vchord CASCADE;
                    EXCEPTION WHEN others THEN
                        PERFORM 1;
                    END;
                    BEGIN
                        CREATE EXTENSION IF NOT EXISTS vectors;
                    EXCEPTION WHEN others THEN
                        PERFORM 1;
                    END;
                    BEGIN
                        CREATE EXTENSION IF NOT EXISTS vector;
                    EXCEPTION WHEN others THEN
                        PERFORM 1;
                    END;
                END $$;
            """)

            # Create sample tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS file (
                    id BIGSERIAL PRIMARY KEY,
                    type VARCHAR(255) NOT NULL,
                    path VARCHAR(255) NOT NULL
                );
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document (
                    id BIGSERIAL PRIMARY KEY,
                    filepath BIGINT REFERENCES file(id),
                    filename TEXT,
                    author TEXT,
                    title TEXT
                );
            """)

            # Insert sample data
            cursor.execute("""
                INSERT INTO file (type, path) VALUES
                ('raw', '/data/doc1.pdf'),
                ('raw', '/data/doc2.pdf'),
                ('image', '/data/img1.png');
            """)

            cursor.execute("""
                INSERT INTO document (filepath, filename, author, title) VALUES
                (1, 'doc1.pdf', 'Author A', 'Title A'),
                (2, 'doc2.pdf', 'Author B', 'Title B');
            """)

        conn.commit()

    yield SOURCE_DB

    # Cleanup: drop source database
    drop_database(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=SOURCE_DB,
        force=True,
    )


@pytest.fixture
def dump_file(source_database):
    """Create a dump file from the source database.

    Yields the path to the dump file.
    Removes the file after the test completes.
    """
    with tempfile.NamedTemporaryFile(suffix=".dump", delete=False) as f:
        dump_path = Path(f.name)

    # Run pg_dump using the shell script
    script_path = Path(__file__).parent.parent.parent / "scripts" / "dump_postgres.sh"

    subprocess.run(  # noqa: S603
        [
            str(script_path),
            "--host",
            DB_HOST,
            "--port",
            str(DB_PORT),
            "--user",
            DB_USER,
            "--password",
            DB_PASSWORD,
            "--dbname",
            source_database,
            "--output",
            str(dump_path),
        ],
        check=True,
        capture_output=True,
    )

    yield dump_path

    # Cleanup: remove dump file
    if dump_path.exists():
        dump_path.unlink()


@pytest.fixture
def target_database():
    """Create an empty target database for restoration.

    Yields the database name after creation.
    Drops the database after the test completes.
    """
    # Create the target database
    create_database(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=TARGET_DB,
    )

    yield TARGET_DB

    # Cleanup: drop target database
    drop_database(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=TARGET_DB,
        force=True,
    )


def _get_table_data(database: str, table: str) -> list[tuple]:
    """Helper to fetch all data from a table."""
    with (
        psycopg.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            dbname=database,
        ) as conn,
        conn.cursor() as cursor,
    ):
        cursor.execute(f"SELECT * FROM {table} ORDER BY id")  # noqa: S608
        return cursor.fetchall()


def _get_table_names(database: str) -> list[str]:
    """Helper to get all table names in a database."""
    with (
        psycopg.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            dbname=database,
        ) as conn,
        conn.cursor() as cursor,
    ):
        cursor.execute("""
                SELECT tablename FROM pg_tables
                WHERE schemaname = 'public'
                ORDER BY tablename;
            """)
        return [row[0] for row in cursor.fetchall()]


@pytest.mark.ci_skip
def test_restore_database(source_database, dump_file, target_database):
    """Test that restore_database correctly restores a database from a dump file.

    Verifies that:
    1. The restore completes successfully
    2. All tables are created in the target database
    3. All data matches between source and target databases
    """
    # Restore the database
    restore_database(
        dump_file=dump_file,
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=target_database,
        port=DB_PORT,
        install_extensions=True,
    )

    # Verify tables exist in target
    source_tables = _get_table_names(source_database)
    target_tables = _get_table_names(target_database)
    assert source_tables == target_tables, "Tables should match between databases"

    # Verify data in each table
    for table in source_tables:
        source_data = _get_table_data(source_database, table)
        target_data = _get_table_data(target_database, table)
        assert source_data == target_data, f"Data in table '{table}' should match"


def test_restore_database_file_not_found():
    """Test that restore_database raises FileNotFoundError for missing dump file."""
    with pytest.raises(FileNotFoundError):
        restore_database(
            dump_file="/nonexistent/path/backup.dump",
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database="dummy_db",
            port=DB_PORT,
        )
