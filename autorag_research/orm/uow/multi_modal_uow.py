"""Multi-modal Unit of Work for AutoRAG-Research.

Provides a comprehensive Unit of Work pattern for multi-modal data ingestion,
including File, Document, Page, Caption, Chunk, ImageChunk, Query, and RetrievalRelation.
"""

from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from autorag_research.exceptions import SessionNotSetError
from autorag_research.orm.repository.caption import CaptionRepository
from autorag_research.orm.repository.chunk import ChunkRepository
from autorag_research.orm.repository.document import DocumentRepository
from autorag_research.orm.repository.file import FileRepository
from autorag_research.orm.repository.image_chunk import ImageChunkRepository
from autorag_research.orm.repository.page import PageRepository
from autorag_research.orm.repository.query import QueryRepository
from autorag_research.orm.repository.retrieval_relation import RetrievalRelationRepository


class MultiModalUnitOfWork:
    """Multi-modal Unit of Work for managing comprehensive data ingestion transactions.

    This UoW includes all entity repositories needed for multi-modal RAG data:
    - File, Document, Page, Caption, Chunk, ImageChunk, Query, RetrievalRelation

    Provides lazy-initialized repositories for efficient resource usage.

    Example:
        ```python
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        engine = create_engine("postgresql://user:pass@localhost/dbname")
        session_factory = sessionmaker(bind=engine)

        with MultiModalUnitOfWork(session_factory) as uow:
            # Access repositories
            file = uow.files.add(File(path="/path/to/image.jpg", type="image"))
            page = uow.pages.add(Page(document_id=1, page_num=1, image_path=file.id))
            uow.commit()
        ```
    """

    def __init__(self, session_factory: sessionmaker[Session], schema: Any | None = None):
        """Initialize Multi-modal Unit of Work with a session factory.

        Args:
            session_factory: SQLAlchemy sessionmaker instance.
            schema: Schema namespace from create_schema(). If None, uses default 768-dim schema.
        """
        self.session_factory = session_factory
        self._schema = schema
        self.session: Session | None = None

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
        # Use default schema
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

    def __enter__(self) -> "MultiModalUnitOfWork":
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
        if self.session is None:
            raise SessionNotSetError
        if self._file_repo is None:
            classes = self._get_schema_classes()
            self._file_repo = FileRepository(self.session, classes["File"])
        return self._file_repo

    @property
    def documents(self) -> DocumentRepository:
        """Get the Document repository.

        Returns:
            DocumentRepository instance.

        Raises:
            SessionNotSetError: If session is not initialized.
        """
        if self.session is None:
            raise SessionNotSetError
        if self._document_repo is None:
            classes = self._get_schema_classes()
            self._document_repo = DocumentRepository(self.session, classes["Document"])
        return self._document_repo

    @property
    def pages(self) -> PageRepository:
        """Get the Page repository.

        Returns:
            PageRepository instance.

        Raises:
            SessionNotSetError: If session is not initialized.
        """
        if self.session is None:
            raise SessionNotSetError
        if self._page_repo is None:
            classes = self._get_schema_classes()
            self._page_repo = PageRepository(self.session, classes["Page"])
        return self._page_repo

    @property
    def captions(self) -> CaptionRepository:
        """Get the Caption repository.

        Returns:
            CaptionRepository instance.

        Raises:
            SessionNotSetError: If session is not initialized.
        """
        if self.session is None:
            raise SessionNotSetError
        if self._caption_repo is None:
            classes = self._get_schema_classes()
            self._caption_repo = CaptionRepository(self.session, classes["Caption"])
        return self._caption_repo

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
            classes = self._get_schema_classes()
            self._chunk_repo = ChunkRepository(self.session, classes["Chunk"])
        return self._chunk_repo

    @property
    def image_chunks(self) -> ImageChunkRepository:
        """Get the ImageChunk repository.

        Returns:
            ImageChunkRepository instance.

        Raises:
            SessionNotSetError: If session is not initialized.
        """
        if self.session is None:
            raise SessionNotSetError
        if self._image_chunk_repo is None:
            classes = self._get_schema_classes()
            self._image_chunk_repo = ImageChunkRepository(self.session, classes["ImageChunk"])
        return self._image_chunk_repo

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
            classes = self._get_schema_classes()
            self._query_repo = QueryRepository(self.session, classes["Query"])
        return self._query_repo

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
            classes = self._get_schema_classes()
            self._retrieval_relation_repo = RetrievalRelationRepository(self.session, classes["RetrievalRelation"])
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
