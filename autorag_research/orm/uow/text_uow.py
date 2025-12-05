"""Text-only Unit of Work for AutoRAG-Research.

Provides a specialized Unit of Work pattern for text-only data ingestion,
focusing on Query, Chunk, and RetrievalRelation repositories.
Image-related tables are not included in this UoW.
"""

from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from autorag_research.exceptions import SessionNotSetError
from autorag_research.orm.repository.chunk import ChunkRepository
from autorag_research.orm.repository.query import QueryRepository
from autorag_research.orm.repository.retrieval_relation import RetrievalRelationRepository


class TextOnlyUnitOfWork:
    """Text-only Unit of Work for managing text data ingestion transactions.

    This UoW focuses on text-based entities only (Query, Chunk, RetrievalRelation)
    and excludes image-related tables like ImageChunk.

    Provides lazy-initialized repositories for efficient resource usage.
    """

    def __init__(self, session_factory: sessionmaker[Session], schema: Any | None = None):
        """Initialize Text-only Unit of Work with a session factory.

        Args:
            session_factory: SQLAlchemy sessionmaker instance.
            schema: Schema namespace from create_schema(). If None, uses default 768-dim schema.
        """
        self.session_factory = session_factory
        self._schema = schema
        self.session: Session | None = None
        self._query_repo: QueryRepository | None = None
        self._chunk_repo: ChunkRepository | None = None
        self._retrieval_relation_repo: RetrievalRelationRepository | None = None

    def _get_schema_classes(self) -> tuple[type, type, type]:
        """Get Query, Chunk, RetrievalRelation classes from schema.

        Returns:
            Tuple of (Query, Chunk, RetrievalRelation) model classes.
        """
        if self._schema is not None:
            return self._schema.Query, self._schema.Chunk, self._schema.RetrievalRelation
        # Use default schema
        from autorag_research.orm.schema import Chunk, Query, RetrievalRelation

        return Query, Chunk, RetrievalRelation

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
            query_cls, _, _ = self._get_schema_classes()
            self._query_repo = QueryRepository(self.session, query_cls)
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
            _, chunk_cls, _ = self._get_schema_classes()
            self._chunk_repo = ChunkRepository(self.session, chunk_cls)
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
            _, _, retrieval_relation_cls = self._get_schema_classes()
            self._retrieval_relation_repo = RetrievalRelationRepository(self.session, retrieval_relation_cls)
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
