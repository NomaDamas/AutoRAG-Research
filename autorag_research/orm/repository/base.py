"""Repository layer for AutoRAG-Research.

Implements Generic Repository + Unit of Work patterns for efficient
CRUD operations and transaction management with SQLAlchemy.
"""

from contextlib import contextmanager
from typing import Any, Generic, TypeVar

from pgvector.sqlalchemy import Vector
from sqlalchemy import select
from sqlalchemy.orm import Session

from autorag_research.exceptions import NoSessionError

T = TypeVar("T")


class GenericRepository(Generic[T]):
    """Generic repository implementing common CRUD operations.

    This base class provides reusable database operations that can be
    extended by specific repositories for custom business logic.
    """

    def __init__(self, session: Session, model_cls: type[T]):
        """Initialize repository with a session and model class.

        Args:
            session: SQLAlchemy session for database operations.
            model_cls: The SQLAlchemy model class this repository manages.
        """
        self.session = session
        self.model_cls = model_cls

    def add(self, entity: T) -> T:
        """Add a new entity to the session.

        Args:
            entity: The entity instance to add.

        Returns:
            The added entity.
        """
        self.session.add(entity)
        return entity

    def add_all(self, entities: list[T]) -> list[T]:
        """Add multiple entities to the session.

        Args:
            entities: List of entity instances to add.

        Returns:
            The added entities.
        """
        self.session.add_all(entities)
        return entities

    def get_by_id(self, _id: Any) -> T | None:
        """Retrieve an entity by its primary key.

        Args:
            _id: The primary key value.

        Returns:
            The entity if found, None otherwise.
        """
        return self.session.get(self.model_cls, _id)

    def get_all(self, limit: int | None = None, offset: int | None = None) -> list[T]:
        """Retrieve all entities of this type.

        Args:
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of all entities.
        """
        stmt = select(self.model_cls)
        if offset:
            stmt = stmt.offset(offset)
        if limit:
            stmt = stmt.limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def update(self, entity: T) -> T:
        """Update an existing entity.

        Args:
            entity: The entity instance to update.

        Returns:
            The updated entity.
        """
        return self.session.merge(entity)

    def delete(self, entity: T) -> None:
        """Delete an entity from the database.

        Args:
            entity: The entity instance to delete.
        """
        self.session.delete(entity)

    def delete_by_id(self, _id: Any) -> bool:
        """Delete an entity by its primary key.

        Args:
            _id: The primary key value.

        Returns:
            True if entity was deleted, False if not found.
        """
        entity = self.get_by_id(_id)
        if entity:
            self.delete(entity)
            return True
        return False

    def count(self) -> int:
        """Count total number of entities.

        Returns:
            Total count of entities.
        """
        return self.session.query(self.model_cls).count()

    def exists(self, _id: Any) -> bool:
        """Check if an entity exists by its primary key.

        Args:
            _id: The primary key value.

        Returns:
            True if entity exists, False otherwise.
        """
        return self.get_by_id(_id) is not None


class UnitOfWork:
    """Unit of Work pattern for managing database transactions.

    Ensures data consistency by grouping multiple repository operations
    into a single atomic transaction.
    """

    def __init__(self, session_factory: Any):
        """Initialize Unit of Work with a session factory.

        Args:
            session_factory: SQLAlchemy sessionmaker instance.
        """
        self.session_factory = session_factory
        self.session: Session | None = None

    def __enter__(self) -> "UnitOfWork":
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


class BaseVectorRepository(GenericRepository[T]):
    """Base repository with vector search capabilities.

    Extends GenericRepository with vector search methods for use with
    pgvector and VectorChord for efficient similarity search.
    """

    def vector_search(
        self,
        query_vector: list[float],
        vector_column: str = "embedding",
        limit: int = 10,
        distance_threshold: float | None = None,
    ) -> list[T]:
        """Perform vector similarity search using VectorChord.

        Args:
            query_vector: The query embedding vector.
            vector_column: Name of the vector column to search.
            limit: Maximum number of results to return.
            distance_threshold: Optional maximum distance threshold.

        Returns:
            List of entities ordered by similarity.

        Note:
            Uses pgvector's cosine distance operator (<=>).
            Requires VectorChord index on embedding column for performance.
        """
        # Convert list to pgvector format
        query_embedding = Vector(query_vector)

        # Get the vector column dynamically
        vector_col = getattr(self.model_cls, vector_column)

        # Build query with distance ordering
        stmt = (
            select(self.model_cls).where(vector_col.is_not(None)).order_by(vector_col.cosine_distance(query_embedding))
        )

        # Apply distance threshold if provided
        if distance_threshold is not None:
            stmt = stmt.where(vector_col.cosine_distance(query_embedding) <= distance_threshold)

        # Apply limit
        stmt = stmt.limit(limit)

        return list(self.session.execute(stmt).scalars().all())

    def vector_search_with_scores(
        self,
        query_vector: list[float],
        vector_column: str = "embedding",
        limit: int = 10,
        distance_threshold: float | None = None,
    ) -> list[tuple[T, float]]:
        """Perform vector similarity search and return entities with their distance scores.

        Args:
            query_vector: The query embedding vector.
            vector_column: Name of the vector column to search.
            limit: Maximum number of results to return.
            distance_threshold: Optional maximum distance threshold.

        Returns:
            List of tuples (entity, distance_score) ordered by similarity.

        Note:
            Lower distance scores indicate higher similarity.
        """
        # Convert list to pgvector format
        query_embedding = Vector(query_vector)

        # Get the vector column dynamically
        vector_col = getattr(self.model_cls, vector_column)

        # Build query with distance as a column
        distance = vector_col.cosine_distance(query_embedding).label("distance")
        stmt = select(self.model_cls, distance).where(vector_col.is_not(None)).order_by(distance)

        # Apply distance threshold if provided
        if distance_threshold is not None:
            stmt = stmt.where(distance <= distance_threshold)

        # Apply limit
        stmt = stmt.limit(limit)

        results = self.session.execute(stmt).all()
        return [(entity, float(dist)) for entity, dist in results]


class BaseEmbeddingRepository(GenericRepository[T]):
    """Base repository with embedding-specific operations.
    This base class is made for schemas that have 'embedding' and 'embeddings' columns.
    """

    def __execute_with_offset_limit(self, stmt: Any, limit: int | None = None, offset: int | None = None) -> list[T]:
        """Helper to execute a statement with optional offset and limit."""
        if offset:
            stmt = stmt.offset(offset)
        if limit:
            stmt = stmt.limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def get_without_embeddings(self, limit: int | None = None, offset: int | None = None) -> list[T]:
        """Retrieve entities that do not have embeddings.

        Args:
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of entities without embeddings.
        """
        stmt = select(self.model_cls).where(self.model_cls.embedding.is_(None))
        return self.__execute_with_offset_limit(stmt, limit, offset)

    def get_with_embeddings(self, limit: int | None = None, offset: int | None = None) -> list[T]:
        """Retrieve entities that have embeddings.

        Args:
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of entities with embeddings.
        """
        stmt = select(self.model_cls).where(self.model_cls.embedding.is_not(None))
        return self.__execute_with_offset_limit(stmt, limit, offset)

    def get_without_multi_embeddings(self, limit: int | None = None, offset: int | None = None) -> list[T]:
        """Retrieve entities that do not have multi-vector embeddings.

        Args:
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of entities without multi-vector embeddings.
        """
        stmt = select(self.model_cls).where(self.model_cls.embeddings.is_(None))
        return self.__execute_with_offset_limit(stmt, limit, offset)

    def get_with_multi_embeddings(self, limit: int | None = None, offset: int | None = None) -> list[T]:
        """Retrieve entities that have multi-vector embeddings.

        Args:
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of entities with multi-vector embeddings.
        """
        stmt = select(self.model_cls).where(self.model_cls.embeddings.is_not(None))
        return self.__execute_with_offset_limit(stmt, limit, offset)


def create_repository(session: Session, model_cls: type[T]) -> GenericRepository[T]:
    """Factory function to create a repository instance.

    Args:
        session: SQLAlchemy session.
        model_cls: The model class for the repository.

    Returns:
        A new GenericRepository instance.

    Example:
        >>> session = SessionFactory()
        >>> user_repo = create_repository(session, User)
        >>> user = user_repo.get_by_id(1)
    """
    return GenericRepository(session, model_cls)


@contextmanager
def repository_context(session_factory, model_cls: type[T]):
    """Context manager for quick repository operations.

    Combines UnitOfWork and Repository creation for simple use cases
    where you need to perform operations on a single model type.

    Args:
        session_factory: SQLAlchemy sessionmaker.
        model_cls: The model class for the repository.

    Yields:
        A tuple of (repository, unit_of_work) for operations.

    Example:
        >>> with repository_context(SessionFactory, User) as (repo, uow):
        ...     user = repo.get_by_id(1)
        ...     user.name = "New Name"
        ...     uow.commit()
    """
    with UnitOfWork(session_factory) as uow:
        if uow.session is None:
            raise NoSessionError
        repo = GenericRepository(uow.session, model_cls)
        yield repo, uow
