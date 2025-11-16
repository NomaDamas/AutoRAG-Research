"""Page repository for AutoRAG-Research.

Implements page-specific CRUD operations and queries extending
the generic repository pattern.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from autorag_research.orm.repository.base import GenericRepository
from autorag_research.orm.schema import Page


class PageRepository(GenericRepository[Page]):
    """Repository for Page entity with specialized queries."""

    def __init__(self, session: Session):
        """Initialize page repository.

        Args:
            session: SQLAlchemy session for database operations.
        """
        super().__init__(session, Page)

    def get_by_document_id(self, document_id: int) -> list[Page]:
        """Retrieve all pages for a specific document.

        Args:
            document_id: The document ID.

        Returns:
            List of pages belonging to the document.
        """
        stmt = select(Page).where(Page.document_id == document_id).order_by(Page.page_num)
        return list(self.session.execute(stmt).scalars().all())

    def get_by_document_and_page_num(self, document_id: int, page_num: int) -> Page | None:
        """Retrieve a specific page by document ID and page number.

        Args:
            document_id: The document ID.
            page_num: The page number.

        Returns:
            The page if found, None otherwise.
        """
        stmt = select(Page).where(Page.document_id == document_id, Page.page_num == page_num)
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def get_with_document(self, page_id: int) -> Page | None:
        """Retrieve a page with its document eagerly loaded.

        Args:
            page_id: The page ID.

        Returns:
            The page with document loaded, None if not found.
        """
        stmt = select(Page).where(Page.id == page_id).options(joinedload(Page.document))
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def get_with_captions(self, page_id: int) -> Page | None:
        """Retrieve a page with its captions eagerly loaded.

        Args:
            page_id: The page ID.

        Returns:
            The page with captions loaded, None if not found.
        """
        stmt = select(Page).where(Page.id == page_id).options(joinedload(Page.captions))
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def get_with_image_chunks(self, page_id: int) -> Page | None:
        """Retrieve a page with its image chunks eagerly loaded.

        Args:
            page_id: The page ID.

        Returns:
            The page with image chunks loaded, None if not found.
        """
        stmt = select(Page).where(Page.id == page_id).options(joinedload(Page.image_chunks))
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def get_with_image_file(self, page_id: int) -> Page | None:
        """Retrieve a page with its image file eagerly loaded.

        Args:
            page_id: The page ID.

        Returns:
            The page with image file loaded, None if not found.
        """
        stmt = select(Page).where(Page.id == page_id).options(joinedload(Page.image_file))
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def get_all_with_document(self, limit: int | None = None, offset: int | None = None) -> list[Page]:
        """Retrieve all pages with their documents eagerly loaded.

        Args:
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of pages with documents loaded.
        """
        stmt = select(Page).options(joinedload(Page.document)).order_by(Page.document_id, Page.page_num)
        if offset:
            stmt = stmt.offset(offset)
        if limit:
            stmt = stmt.limit(limit)
        return list(self.session.execute(stmt).scalars().unique().all())

    def search_by_metadata(self, metadata_key: str, metadata_value: str) -> list[Page]:
        """Search pages by metadata field.

        Args:
            metadata_key: The key in the JSONB metadata field.
            metadata_value: The value to search for.

        Returns:
            List of matching pages.
        """
        stmt = select(Page).where(Page.page_metadata[metadata_key].astext == metadata_value)
        return list(self.session.execute(stmt).scalars().all())

    def get_by_image_path_id(self, image_path_id: int) -> Page | None:
        """Retrieve a page by its image path ID.

        Args:
            image_path_id: The file ID for the image.

        Returns:
            The page if found, None otherwise.
        """
        stmt = select(Page).where(Page.image_path == image_path_id)
        return self.session.execute(stmt).scalar_one_or_none()

    def count_by_document(self, document_id: int) -> int:
        """Count the number of pages in a document.

        Args:
            document_id: The document ID.

        Returns:
            Number of pages in the document.
        """
        return self.session.query(Page).filter(Page.document_id == document_id).count()

    def get_page_range(self, document_id: int, start_page: int, end_page: int) -> list[Page]:
        """Retrieve a range of pages from a document.

        Args:
            document_id: The document ID.
            start_page: The starting page number (inclusive).
            end_page: The ending page number (inclusive).

        Returns:
            List of pages in the specified range.
        """
        stmt = (
            select(Page)
            .where(Page.document_id == document_id, Page.page_num >= start_page, Page.page_num <= end_page)
            .order_by(Page.page_num)
        )
        return list(self.session.execute(stmt).scalars().all())
