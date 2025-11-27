"""Text-only Unit of Work for AutoRAG-Research.

Provides a specialized Unit of Work pattern for text-only data ingestion,
focusing on Query, Chunk, and RetrievalRelation repositories.
Image-related tables are not included in this UoW.
"""

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.exceptions import SessionNotSetError
from autorag_research.orm.repository.base import GenericRepository
from autorag_research.orm.repository.chunk import ChunkRepository
from autorag_research.orm.repository.query import QueryRepository
from autorag_research.orm.schema import RetrievalRelation


class RetrievalRelationRepository(GenericRepository[RetrievalRelation]):
    """Repository for RetrievalRelation entity (text-only, no image_chunk support)."""

    def __init__(self, session: Session):
        """Initialize retrieval relation repository.

        Args:
            session: SQLAlchemy session for database operations.
        """
        super().__init__(session, RetrievalRelation)

    def get_by_query_id(self, query_id: int) -> list[RetrievalRelation]:
        """Retrieve all retrieval relations for a specific query.

        Args:
            query_id: The query ID.

        Returns:
            List of retrieval relations for the query, ordered by group_index and group_order.
        """
        stmt = (
            select(RetrievalRelation)
            .where(RetrievalRelation.query_id == query_id)
            .order_by(RetrievalRelation.group_index, RetrievalRelation.group_order)
        )
        return list(self.session.execute(stmt).scalars().all())

    def get_max_group_index(self, query_id: int) -> int | None:
        """Get the maximum group_index for a query.

        Args:
            query_id: The query ID.

        Returns:
            Maximum group_index value, None if no relations exist.
        """
        stmt = select(func.max(RetrievalRelation.group_index)).where(RetrievalRelation.query_id == query_id)
        return self.session.execute(stmt).scalar_one_or_none()

    def get_max_group_order(self, query_id: int, group_index: int) -> int | None:
        """Get the maximum group_order for a specific group.

        Args:
            query_id: The query ID.
            group_index: The group index.

        Returns:
            Maximum group_order value, None if no relations exist.
        """
        stmt = select(func.max(RetrievalRelation.group_order)).where(
            RetrievalRelation.query_id == query_id,
            RetrievalRelation.group_index == group_index,
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def count_by_query(self, query_id: int) -> int:
        """Count the number of retrieval relations for a query.

        Args:
            query_id: The query ID.

        Returns:
            Number of retrieval relations for the query.
        """
        stmt = select(func.count()).select_from(RetrievalRelation).where(RetrievalRelation.query_id == query_id)
        return self.session.execute(stmt).scalar_one()


class TextOnlyUnitOfWork:
    """Text-only Unit of Work for managing text data ingestion transactions.

    This UoW focuses on text-based entities only (Query, Chunk, RetrievalRelation)
    and excludes image-related tables like ImageChunk.

    Provides lazy-initialized repositories for efficient resource usage.
    """

    def __init__(self, session_factory: sessionmaker[Session]):
        """Initialize Text-only Unit of Work with a session factory.

        Args:
            session_factory: SQLAlchemy sessionmaker instance.
        """
        self.session_factory = session_factory
        self.session: Session | None = None
        self._query_repo: QueryRepository | None = None
        self._chunk_repo: ChunkRepository | None = None
        self._retrieval_relation_repo: RetrievalRelationRepository | None = None

    def __enter__(self) -> "TextOnlyUnitOfWork":
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
        # Reset repository references
        self._query_repo = None
        self._chunk_repo = None
        self._retrieval_relation_repo = None

    @property
    def queries(self) -> QueryRepository:
        """Get the Query repository.

        Returns:
            QueryRepository instance.

        Raises:
            SessionNotSetError: If session is not initialized.
        """
        if self.session is None:
            raise SessionNotSetError
        if self._query_repo is None:
            self._query_repo = QueryRepository(self.session)
        return self._query_repo

    @property
    def chunks(self) -> ChunkRepository:
        """Get the Chunk repository.

        Returns:
            ChunkRepository instance.

        Raises:
            SessionNotSetError: If session is not initialized.
        """
        if self.session is None:
            raise SessionNotSetError
        if self._chunk_repo is None:
            self._chunk_repo = ChunkRepository(self.session)
        return self._chunk_repo

    @property
    def retrieval_relations(self) -> RetrievalRelationRepository:
        """Get the RetrievalRelation repository.

        Returns:
            RetrievalRelationRepository instance.

        Raises:
            SessionNotSetError: If session is not initialized.
        """
        if self.session is None:
            raise SessionNotSetError
        if self._retrieval_relation_repo is None:
            self._retrieval_relation_repo = RetrievalRelationRepository(self.session)
        return self._retrieval_relation_repo

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
