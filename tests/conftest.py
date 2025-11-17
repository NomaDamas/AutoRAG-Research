import os
from collections.abc import Generator
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, scoped_session, sessionmaker

from autorag_research.exceptions import EnvNotFoundError


@pytest.fixture(scope="session")
def db_engine():
    """Create a database engine for the test session.

    Expects POSTGRES_URL environment variable to be set.
    The database, tables, and data should already exist at this URL.
    """
    postgres_url = os.getenv("POSTGRES_URL")
    if not postgres_url:
        raise EnvNotFoundError("POSTGRES_URL")

    engine = create_engine(
        postgres_url,
        pool_pre_ping=True,  # Check connection health
        pool_size=10,
        max_overflow=20,
    )

    yield engine

    engine.dispose()


@pytest.fixture(scope="session")
def session_factory(db_engine):
    """Create a thread-safe scoped session factory for the test session.

    This follows the pattern from db_pattern.md using scoped_session.
    The scoped_session ensures thread-safety for multi-worker test execution.
    """
    return scoped_session(sessionmaker(bind=db_engine))


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
