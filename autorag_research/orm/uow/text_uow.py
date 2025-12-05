"""Text-only Unit of Work for AutoRAG-Research.

Provides a specialized Unit of Work pattern for text-only data ingestion,
focusing on Query, Chunk, and RetrievalRelation repositories.
Image-related tables are not included in this UoW.
"""

from typing import Any

from sqlalchemy.orm import sessionmaker

from autorag_research.orm.repository.chunk import ChunkRepository
from autorag_research.orm.repository.query import QueryRepository
from autorag_research.orm.repository.retrieval_relation import RetrievalRelationRepository
from autorag_research.orm.uow.base import BaseUnitOfWork


class TextOnlyUnitOfWork(BaseUnitOfWork):
    """Text-only Unit of Work for managing text data ingestion transactions.

    This UoW focuses on text-based entities only (Query, Chunk, RetrievalRelation)
    and excludes image-related tables like ImageChunk.

    Provides lazy-initialized repositories for efficient resource usage.
    """

    def __init__(self, session_factory: sessionmaker, schema: Any | None = None):
        """Initialize Text-only Unit of Work with a session factory.

        Args:
            session_factory: SQLAlchemy sessionmaker instance.
            schema: Schema namespace from create_schema(). If None, uses default 768-dim schema.
        """
        super().__init__(session_factory, schema)
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

        from autorag_research.orm.schema import Chunk, Query, RetrievalRelation

        return Query, Chunk, RetrievalRelation

    def _reset_repositories(self) -> None:
        """Reset all repository references to None."""
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
        return self._get_repository(
            "_query_repo",
            QueryRepository,
            lambda: self._get_schema_classes()[0],
        )

    @property
    def chunks(self) -> ChunkRepository:
        """Get the Chunk repository.

        Returns:
            ChunkRepository instance.

        Raises:
            SessionNotSetError: If session is not initialized.
        """
        return self._get_repository(
            "_chunk_repo",
            ChunkRepository,
            lambda: self._get_schema_classes()[1],
        )

    @property
    def retrieval_relations(self) -> RetrievalRelationRepository:
        """Get the RetrievalRelation repository.

        Returns:
            RetrievalRelationRepository instance.

        Raises:
            SessionNotSetError: If session is not initialized.
        """
        return self._get_repository(
            "_retrieval_relation_repo",
            RetrievalRelationRepository,
            lambda: self._get_schema_classes()[2],
        )
