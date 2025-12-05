"""Base Unit of Work for AutoRAG-Research.

Provides abstract base class with common UoW patterns to reduce duplication
across TextOnlyUnitOfWork, MultiModalUnitOfWork, and RetrievalUnitOfWork.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from sqlalchemy.orm import Session, sessionmaker
from typing_extensions import Self

from autorag_research.exceptions import SessionNotSetError


class BaseUnitOfWork(ABC):
    """Abstract base class for Unit of Work pattern.

    Provides common functionality for managing database transactions:
    - Session lifecycle management (context manager)
    - Transaction operations (commit, rollback, flush)
    - Lazy repository initialization helper

    Subclasses must implement:
    - `_get_schema_classes()`: Return schema classes needed for repositories
    - Repository properties using `_get_repository()` helper
    """

    def __init__(self, session_factory: sessionmaker[Session], schema: Any | None = None):
        """Initialize Unit of Work with session factory and optional schema.

        Args:
            session_factory: SQLAlchemy sessionmaker instance.
            schema: Schema namespace from create_schema(). If None, uses default 768-dim schema.
        """
        self.session_factory = session_factory
        self._schema = schema
        self.session: Session | None = None

    @abstractmethod
    def _get_schema_classes(self) -> Any:
        """Get model classes from schema.

        Returns:
            Schema classes (implementation-specific: tuple, dict, or other structure).
        """
        ...

    def __enter__(self) -> Self:
        """Enter the context manager and create a new session.

        Returns:
            Self for method chaining.
        """
        self.session = self.session_factory()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context manager and clean up session.

        Automatically rolls back if an exception occurred.

        Args:
            exc_type: Exception type if an error occurred.
            exc_val: Exception value if an error occurred.
            exc_tb: Exception traceback if an error occurred.
        """
        if exc_type is not None:
            self.rollback()
        if self.session:
            self.session.close()
            self.session = None
        self._reset_repositories()

    @abstractmethod
    def _reset_repositories(self) -> None:
        """Reset all repository references to None.

        Called during cleanup to ensure repositories are recreated on next access.
        Subclasses must implement this to reset their specific repository attributes.
        """
        ...

    def _get_repository(
        self,
        repo_attr: str,
        repo_class: type,
        schema_class_getter: Callable[[], type],
    ) -> Any:
        """Helper method for lazy repository initialization.

        Args:
            repo_attr: Name of the private repository attribute (e.g., "_query_repo").
            repo_class: Repository class to instantiate.
            schema_class_getter: Callable that returns the schema model class.

        Returns:
            Repository instance.

        Raises:
            SessionNotSetError: If session is not initialized.
        """
        if self.session is None:
            raise SessionNotSetError

        # Check if repository is already cached
        cached_repo = getattr(self, repo_attr, None)
        if cached_repo is not None:
            return cached_repo

        # Create new repository instance
        schema_class = schema_class_getter()
        repo = repo_class(self.session, schema_class)
        setattr(self, repo_attr, repo)
        return repo

    def commit(self) -> None:
        """Commit the current transaction."""
        if self.session:
            self.session.commit()

    def rollback(self) -> None:
        """Rollback the current transaction."""
        if self.session:
            self.session.rollback()

    def flush(self) -> None:
        """Flush pending changes without committing."""
        if self.session:
            self.session.flush()
