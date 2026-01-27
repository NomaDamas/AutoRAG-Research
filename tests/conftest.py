import json
import shutil
import subprocess
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv
from sqlalchemy.orm import Session

from autorag_research.orm.connection import DBConnection
from autorag_research.orm.schema import Chunk

# Load environment variables from postgresql/.env
_env_path = Path(__file__).parent.parent / "postgresql" / ".env"
load_dotenv(_env_path)


@pytest.fixture(scope="session")
def db_connection() -> DBConnection:
    """Create a DBConnection instance for the test session.

    Reads configuration from postgresql/.env file.
    """
    conn = DBConnection.from_env()
    return conn


@pytest.fixture(scope="session")
def db_engine(db_connection):
    """Create a database engine for the test session.

    Reads configuration from postgresql/.env file.
    """
    engine = db_connection.get_engine()

    yield engine

    engine.dispose()


@pytest.fixture(scope="session")
def session_factory(db_connection):
    """Create a thread-safe scoped session factory for the test session.

    This follows the pattern from db_pattern.md using scoped_session.
    The scoped_session ensures thread-safety for multi-worker test execution.
    """
    return db_connection.get_scoped_session_factory()


@pytest.fixture
def db_session(session_factory) -> Generator[Session, Any, None]:
    """Create a new database session for each test.

    This fixture provides a clean, thread-safe session for each test function.
    The session is automatically rolled back after the test to maintain isolation.
    """
    session = session_factory()

    yield session

    session.rollback()
    session_factory.remove()  # Clean up the scoped session


@pytest.fixture(scope="session")
def bm25_index_path(session_factory):
    """Create a BM25 index from seed data chunks.

    This fixture:
    1. Fetches chunks from the database (seed data from 002-seed.sql)
    2. Writes them to a JSONL file
    3. Builds a Lucene index using pyserini
    4. Returns the index path

    The index is created once per test session and cleaned up afterward.
    """
    if not shutil.which("java"):
        pytest.fail("Java Development Kit (JDK) not found. Pyserini requires Java 11+.")

    # Fetch chunks from seed data
    session = session_factory()
    try:
        chunks = session.query(Chunk).all()
        if not chunks:
            pytest.skip("No chunks found in database - seed data may not be loaded")
    finally:
        session.close()

    # Create temporary directory for index
    temp_dir = tempfile.mkdtemp(prefix="bm25_test_index_")
    temp_path = Path(temp_dir)
    docs_dir = temp_path / "docs"
    index_dir = temp_path / "index"
    docs_dir.mkdir()

    # Write chunks to JSONL file
    jsonl_file = docs_dir / "chunks.jsonl"
    with open(jsonl_file, "w") as f:
        for chunk in chunks:
            doc = {"id": str(chunk.id), "contents": chunk.contents}
            f.write(json.dumps(doc) + "\n")

    # Build Lucene index using pyserini CLI
    cmd = [
        "python",
        "-m",
        "pyserini.index.lucene",
        "--collection",
        "JsonCollection",
        "--input",
        str(docs_dir),
        "--index",
        str(index_dir),
        "--generator",
        "DefaultLuceneDocumentGenerator",
        "--threads",
        "1",
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        shutil.rmtree(temp_dir)
        pytest.fail(f"Failed to build BM25 index: {result.stderr}")

    yield str(index_dir)

    # Cleanup
    shutil.rmtree(temp_dir)
