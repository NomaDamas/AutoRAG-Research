import os
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, scoped_session, sessionmaker

from autorag_research.exceptions import EnvNotFoundError

# Load environment variables from postgresql/.env
_env_path = Path(__file__).parent.parent / "postgresql" / ".env"
load_dotenv(_env_path)


@pytest.fixture(scope="session")
def db_engine():
    """Create a database engine for the test session.

    Reads configuration from postgresql/.env file.
    """
    host = os.getenv("POSTGRES_HOST", "localhost")
    user = os.getenv("POSTGRES_USER")
    pwd = os.getenv("POSTGRES_PASSWORD")
    port = int(os.getenv("PG_PORT", os.getenv("POSTGRES_PORT", "5432")))
    db_name = os.getenv("TEST_DB_NAME", os.getenv("POSTGRES_DB"))
    if not all([host, user, pwd, port, db_name]):
        raise EnvNotFoundError(  # noqa: TRY003
            "POSTGRES_HOST, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_PORT, and TEST_DB_NAME must be set"
        )

    postgres_url = f"postgresql+psycopg://{user}:{pwd}@{host}:{port}/{db_name}"

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
