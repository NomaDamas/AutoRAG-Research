"""Page repository for AutoRAG-Research.

Implements page-specific CRUD operations and queries extending
the generic repository pattern.
"""

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from autorag_research.orm.repository.base import GenericRepository


class PageRepository(GenericRepository):
    """Repository for Page entity with specialized queries."""

    def __init__(self, session: Session, model_cls: type | None = None):
        """Initialize page repository.

        Args:
            session: SQLAlchemy session for database operations.
            model_cls: The Page model class. If None, uses default schema.
        """
        if model_cls is None:
            from autorag_research.orm.schema import Page

            model_cls = Page
        super().__init__(session, model_cls)

    def get_by_document_id(self, document_id: int | str) -> list[Any]:
        """Retrieve all pages for a specific document.

        Args:
            document_id: The document ID.

        Returns:
            List of pages belonging to the document.
        """
        stmt = select(self.model_cls).where(self.model_cls.document_id == document_id).order_by(self.model_cls.page_num)
        return list(self.session.execute(stmt).scalars().all())

    def get_by_document_and_page_num(self, document_id: int | str, page_num: int) -> Any | None:
        """Retrieve a specific page by document ID and page number.

        Args:
            document_id: The document ID.
            page_num: The page number.

        Returns:
            The page if found, None otherwise.
        """
        stmt = select(self.model_cls).where(
            self.model_cls.document_id == document_id,
            self.model_cls.page_num == page_num,
        )
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def get_with_document(self, page_id: int | str) -> Any | None:
        """Retrieve a page with its document eagerly loaded.

        Args:
            page_id: The page ID.

        Returns:
            The page with document loaded, None if not found.
        """
        stmt = select(self.model_cls).where(self.model_cls.id == page_id).options(joinedload(self.model_cls.document))
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def get_with_captions(self, page_id: int | str) -> Any | None:
        """Retrieve a page with its captions eagerly loaded.

        Args:
            page_id: The page ID.

        Returns:
            The page with captions loaded, None if not found.
        """
        stmt = select(self.model_cls).where(self.model_cls.id == page_id).options(joinedload(self.model_cls.captions))
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def get_with_image_chunks(self, page_id: int | str) -> Any | None:
        """Retrieve a page with its image chunks eagerly loaded.

        Args:
            page_id: The page ID.

        Returns:
            The page with image chunks loaded, None if not found.
        """
        stmt = (
            select(self.model_cls).where(self.model_cls.id == page_id).options(joinedload(self.model_cls.image_chunks))
        )
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def get_all_with_document(self, limit: int | None = None, offset: int | None = None) -> list[Any]:
        """Retrieve all pages with their documents eagerly loaded.

        Args:
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of pages with documents loaded.
        """
        stmt = (
            select(self.model_cls)
            .options(joinedload(self.model_cls.document))
            .order_by(self.model_cls.document_id, self.model_cls.page_num)
        )
        if offset:
            stmt = stmt.offset(offset)
        if limit:
            stmt = stmt.limit(limit)
        return list(self.session.execute(stmt).scalars().unique().all())

    def search_by_metadata(self, metadata_key: str, metadata_value: str) -> list[Any]:
        """Search pages by metadata field.

        Args:
            metadata_key: The key in the JSONB metadata field.
            metadata_value: The value to search for.

        Returns:
            List of matching pages.
        """
        stmt = select(self.model_cls).where(self.model_cls.page_metadata[metadata_key].astext == metadata_value)
        return list(self.session.execute(stmt).scalars().all())

    def count_by_document(self, document_id: int | str) -> int:
        """Count the number of pages in a document.

        Args:
            document_id: The document ID.

        Returns:
            Number of pages in the document.
        """
        return self.session.query(self.model_cls).filter(self.model_cls.document_id == document_id).count()

    def get_page_range(self, document_id: int | str, start_page: int, end_page: int) -> list[Any]:
        """Retrieve a range of pages from a document.

        Args:
            document_id: The document ID.
            start_page: The starting page number (inclusive).
            end_page: The ending page number (inclusive).

        Returns:
            List of pages in the specified range.
        """
        stmt = (
            select(self.model_cls)
            .where(
                self.model_cls.document_id == document_id,
                self.model_cls.page_num >= start_page,
                self.model_cls.page_num <= end_page,
            )
            .order_by(self.model_cls.page_num)
        )
        return list(self.session.execute(stmt).scalars().all())
