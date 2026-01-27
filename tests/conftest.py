from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv
from sqlalchemy.orm import Session

from autorag_research.orm.connection import DBConnection

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
