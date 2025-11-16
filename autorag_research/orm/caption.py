"""Caption repository for AutoRAG-Research.

Implements caption-specific CRUD operations and queries extending
the generic repository pattern.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from autorag_research.orm.repository import GenericRepository
from autorag_research.orm.schema import Caption


class CaptionRepository(GenericRepository[Caption]):
    """Repository for Caption entity with specialized queries."""

    def __init__(self, session: Session):
        """Initialize caption repository.

        Args:
            session: SQLAlchemy session for database operations.
        """
        super().__init__(session, Caption)

    def get_by_page_id(self, page_id: int) -> list[Caption]:
        """Retrieve all captions for a specific page.

        Args:
            page_id: The page ID.

        Returns:
            List of captions belonging to the page.
        """
        stmt = select(Caption).where(Caption.page_id == page_id)
        return list(self.session.execute(stmt).scalars().all())

    def get_with_page(self, caption_id: int) -> Caption | None:
        """Retrieve a caption with its page eagerly loaded.

        Args:
            caption_id: The caption ID.

        Returns:
            The caption with page loaded, None if not found.
        """
        stmt = select(Caption).where(Caption.id == caption_id).options(joinedload(Caption.page))
        return self.session.execute(stmt).scalar_one_or_none()

    def get_with_chunks(self, caption_id: int) -> Caption | None:
        """Retrieve a caption with its chunks eagerly loaded.

        Args:
            caption_id: The caption ID.

        Returns:
            The caption with chunks loaded, None if not found.
        """
        stmt = select(Caption).where(Caption.id == caption_id).options(joinedload(Caption.chunks))
        return self.session.execute(stmt).scalar_one_or_none()

    def get_with_caption_chunk_relations(self, caption_id: int) -> Caption | None:
        """Retrieve a caption with its caption-chunk relations eagerly loaded.

        Args:
            caption_id: The caption ID.

        Returns:
            The caption with caption-chunk relations loaded, None if not found.
        """
        stmt = select(Caption).where(Caption.id == caption_id).options(joinedload(Caption.caption_chunk_relations))
        return self.session.execute(stmt).scalar_one_or_none()

    def search_by_contents(self, search_text: str) -> list[Caption]:
        """Search captions by contents using SQL LIKE.

        Args:
            search_text: The text to search for (use % as wildcard).

        Returns:
            List of matching captions.
        """
        stmt = select(Caption).where(Caption.contents.like(f"%{search_text}%"))
        return list(self.session.execute(stmt).scalars().all())

    def get_all_with_page(self, limit: int | None = None, offset: int | None = None) -> list[Caption]:
        """Retrieve all captions with their pages eagerly loaded.

        Args:
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of captions with pages loaded.
        """
        stmt = select(Caption).options(joinedload(Caption.page))
        if offset:
            stmt = stmt.offset(offset)
        if limit:
            stmt = stmt.limit(limit)
        return list(self.session.execute(stmt).scalars().unique().all())

    def count_by_page(self, page_id: int) -> int:
        """Count the number of captions for a specific page.

        Args:
            page_id: The page ID.

        Returns:
            Number of captions for the page.
        """
        return self.session.query(Caption).filter(Caption.page_id == page_id).count()

    def get_by_contents_exact(self, contents: str) -> list[Caption]:
        """Retrieve captions with exact contents match.

        Args:
            contents: The exact contents to search for.

        Returns:
            List of captions with matching contents.
        """
        stmt = select(Caption).where(Caption.contents == contents)
        return list(self.session.execute(stmt).scalars().all())

    def get_with_all_relations(self, caption_id: int) -> Caption | None:
        """Retrieve a caption with all relationships eagerly loaded.

        Args:
            caption_id: The caption ID.

        Returns:
            The caption with all relations loaded, None if not found.
        """
        stmt = (
            select(Caption)
            .where(Caption.id == caption_id)
            .options(
                joinedload(Caption.page),
                joinedload(Caption.chunks),
                joinedload(Caption.caption_chunk_relations),
            )
        )
        return self.session.execute(stmt).scalar_one_or_none()
