"""Multi-modal Unit of Work for AutoRAG-Research.

Provides a comprehensive Unit of Work pattern for multi-modal data ingestion,
including File, Document, Page, Caption, Chunk, ImageChunk, Query, and RetrievalRelation.
"""

from typing import Any

from sqlalchemy.orm import sessionmaker

from autorag_research.orm.repository.caption import CaptionRepository
from autorag_research.orm.repository.chunk import ChunkRepository
from autorag_research.orm.repository.document import DocumentRepository
from autorag_research.orm.repository.file import FileRepository
from autorag_research.orm.repository.image_chunk import ImageChunkRepository
from autorag_research.orm.repository.page import PageRepository
from autorag_research.orm.repository.query import QueryRepository
from autorag_research.orm.repository.retrieval_relation import RetrievalRelationRepository
from autorag_research.orm.uow.base import BaseUnitOfWork


class MultiModalUnitOfWork(BaseUnitOfWork):
    """Multi-modal Unit of Work for managing comprehensive data ingestion transactions.

    This UoW includes all entity repositories needed for multi-modal RAG data:
    - File, Document, Page, Caption, Chunk, ImageChunk, Query, RetrievalRelation

    Provides lazy-initialized repositories for efficient resource usage.

    Example:
        ```python
        from autorag_research.orm.connection import DBConnection

        db = DBConnection.from_config()  # or DBConnection.from_env()
        session_factory = db.get_session_factory()

        with MultiModalUnitOfWork(session_factory) as uow:
            # Access repositories
            file = uow.files.add(File(path="/path/to/image.jpg", type="image"))
            page = uow.pages.add(Page(document_id=1, page_num=1, image_path=file.id))
            uow.commit()
        ```
    """

    def __init__(self, session_factory: sessionmaker, schema: Any | None = None):
        """Initialize Multi-modal Unit of Work with a session factory.

        Args:
            session_factory: SQLAlchemy sessionmaker instance.
            schema: Schema namespace from create_schema(). If None, uses default 768-dim schema.
        """
        super().__init__(session_factory, schema)

        # Lazy-initialized repositories
        self._file_repo: FileRepository | None = None
        self._document_repo: DocumentRepository | None = None
        self._page_repo: PageRepository | None = None
        self._caption_repo: CaptionRepository | None = None
        self._chunk_repo: ChunkRepository | None = None
        self._image_chunk_repo: ImageChunkRepository | None = None
        self._query_repo: QueryRepository | None = None
        self._retrieval_relation_repo: RetrievalRelationRepository | None = None

    def _get_schema_classes(self) -> dict[str, type]:
        """Get all model classes from schema.

        Returns:
            Dictionary mapping class names to model classes.
        """
        if self._schema is not None:
            return {
                "File": self._schema.File,
                "Document": self._schema.Document,
                "Page": self._schema.Page,
                "Caption": self._schema.Caption,
                "Chunk": self._schema.Chunk,
                "ImageChunk": self._schema.ImageChunk,
                "Query": self._schema.Query,
                "RetrievalRelation": self._schema.RetrievalRelation,
            }

        from autorag_research.orm.schema import (
            Caption,
            Chunk,
            Document,
            File,
            ImageChunk,
            Page,
            Query,
            RetrievalRelation,
        )

        return {
            "File": File,
            "Document": Document,
            "Page": Page,
            "Caption": Caption,
            "Chunk": Chunk,
            "ImageChunk": ImageChunk,
            "Query": Query,
            "RetrievalRelation": RetrievalRelation,
        }

    def _reset_repositories(self) -> None:
        """Reset all repository references to None."""
        self._file_repo = None
        self._document_repo = None
        self._page_repo = None
        self._caption_repo = None
        self._chunk_repo = None
        self._image_chunk_repo = None
        self._query_repo = None
        self._retrieval_relation_repo = None

    @property
    def files(self) -> FileRepository:
        """Get the File repository.

        Returns:
            FileRepository instance.

        Raises:
            SessionNotSetError: If session is not initialized.
        """
        return self._get_repository(
            "_file_repo",
            FileRepository,
            lambda: self._get_schema_classes()["File"],
        )

    @property
    def documents(self) -> DocumentRepository:
        """Get the Document repository.

        Returns:
            DocumentRepository instance.

        Raises:
            SessionNotSetError: If session is not initialized.
        """
        return self._get_repository(
            "_document_repo",
            DocumentRepository,
            lambda: self._get_schema_classes()["Document"],
        )

    @property
    def pages(self) -> PageRepository:
        """Get the Page repository.

        Returns:
            PageRepository instance.

        Raises:
            SessionNotSetError: If session is not initialized.
        """
        return self._get_repository(
            "_page_repo",
            PageRepository,
            lambda: self._get_schema_classes()["Page"],
        )

    @property
    def captions(self) -> CaptionRepository:
        """Get the Caption repository.

        Returns:
            CaptionRepository instance.

        Raises:
            SessionNotSetError: If session is not initialized.
        """
        return self._get_repository(
            "_caption_repo",
            CaptionRepository,
            lambda: self._get_schema_classes()["Caption"],
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
            lambda: self._get_schema_classes()["Chunk"],
        )

    @property
    def image_chunks(self) -> ImageChunkRepository:
        """Get the ImageChunk repository.

        Returns:
            ImageChunkRepository instance.

        Raises:
            SessionNotSetError: If session is not initialized.
        """
        return self._get_repository(
            "_image_chunk_repo",
            ImageChunkRepository,
            lambda: self._get_schema_classes()["ImageChunk"],
        )

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
            lambda: self._get_schema_classes()["Query"],
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
            lambda: self._get_schema_classes()["RetrievalRelation"],
        )
