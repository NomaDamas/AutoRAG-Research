"""Document repository for AutoRAG-Research.

Implements document-specific CRUD operations and queries extending
the generic repository pattern.
"""

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from autorag_research.orm.repository.base import GenericRepository


class DocumentRepository(GenericRepository):
    """Repository for Document entity with specialized queries."""

    def __init__(self, session: Session, model_cls: type | None = None):
        """Initialize document repository.

        Args:
            session: SQLAlchemy session for database operations.
            model_cls: The Document model class. If None, uses default schema.
        """
        if model_cls is None:
            from autorag_research.orm.schema import Document

            model_cls = Document
        super().__init__(session, model_cls)

    def get_by_filename(self, filename: str) -> Any | None:
        """Retrieve a document by its filename.

        Args:
            filename: The filename to search for.

        Returns:
            The document if found, None otherwise.
        """
        stmt = select(self.model_cls).where(self.model_cls.filename == filename)
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def get_by_title(self, title: str) -> Any | None:
        """Retrieve a document by its title.

        Args:
            title: The title to search for.

        Returns:
            The document if found, None otherwise.
        """
        stmt = select(self.model_cls).where(self.model_cls.title == title)
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def get_by_author(self, author: str) -> list[Any]:
        """Retrieve all documents by a specific author.

        Args:
            author: The author name to search for.

        Returns:
            List of documents by the author.
        """
        stmt = select(self.model_cls).where(self.model_cls.author == author)
        return list(self.session.execute(stmt).scalars().all())

    def get_with_pages(self, document_id: int) -> Any | None:
        """Retrieve a document with its pages eagerly loaded.

        Args:
            document_id: The document ID.

        Returns:
            The document with pages loaded, None if not found.
        """
        stmt = select(self.model_cls).where(self.model_cls.id == document_id).options(joinedload(self.model_cls.pages))
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def get_with_file(self, document_id: int) -> Any | None:
        """Retrieve a document with its file eagerly loaded.

        Args:
            document_id: The document ID.

        Returns:
            The document with file loaded, None if not found.
        """
        stmt = select(self.model_cls).where(self.model_cls.id == document_id).options(joinedload(self.model_cls.file))
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def get_all_with_pages(self, limit: int | None = None, offset: int | None = None) -> list[Any]:
        """Retrieve all documents with their pages eagerly loaded.

        Args:
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of documents with pages loaded.
        """
        stmt = select(self.model_cls).options(joinedload(self.model_cls.pages))
        if offset:
            stmt = stmt.offset(offset)
        if limit:
            stmt = stmt.limit(limit)
        return list(self.session.execute(stmt).scalars().unique().all())

    def search_by_metadata(self, metadata_key: str, metadata_value: str) -> list[Any]:
        """Search documents by metadata field.

        Args:
            metadata_key: The key in the JSONB metadata field.
            metadata_value: The value to search for.

        Returns:
            List of matching documents.
        """
        stmt = select(self.model_cls).where(self.model_cls.doc_metadata[metadata_key].astext == metadata_value)
        return list(self.session.execute(stmt).scalars().all())

    def count_pages(self, document_id: int) -> int:
        """Count the number of pages in a document.

        Args:
            document_id: The document ID.

        Returns:
            Number of pages in the document.
        """
        document = self.get_with_pages(document_id)
        return len(document.pages) if document else 0

    def get_by_filepath_id(self, filepath_id: int) -> Any | None:
        """Retrieve a document by its file path ID.

        Args:
            filepath_id: The file ID to search for.

        Returns:
            The document if found, None otherwise.
        """
        stmt = select(self.model_cls).where(self.model_cls.filepath == filepath_id)
        return self.session.execute(stmt).scalar_one_or_none()
