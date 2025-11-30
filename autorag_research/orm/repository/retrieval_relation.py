"""RetrievalRelation repository for AutoRAG-Research.

Supports both text chunks and image chunks for multi-modal retrieval ground truth.
"""

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from autorag_research.orm.repository.base import GenericRepository


class RetrievalRelationRepository(GenericRepository[Any]):
    """Repository for RetrievalRelation entity with multi-modal support.

    This repository handles retrieval ground truth relations that can link
    queries to either text chunks or image chunks (mutually exclusive).
    """

    def __init__(self, session: Session, model_cls: type | None = None):
        """Initialize retrieval relation repository.

        Args:
            session: SQLAlchemy session for database operations.
            model_cls: The RetrievalRelation model class. If None, uses default schema.
        """
        if model_cls is None:
            from autorag_research.orm.schema import RetrievalRelation

            model_cls = RetrievalRelation
        super().__init__(session, model_cls)

    def get_by_query_id(self, query_id: int) -> list[Any]:
        """Retrieve all retrieval relations for a specific query.

        Args:
            query_id: The query ID.

        Returns:
            List of retrieval relations for the query, ordered by group_index and group_order.
        """
        stmt = (
            select(self.model_cls)
            .where(self.model_cls.query_id == query_id)
            .order_by(self.model_cls.group_index, self.model_cls.group_order)
        )
        return list(self.session.execute(stmt).scalars().all())

    def get_by_query_and_chunk(self, query_id: int, chunk_id: int) -> Any | None:
        """Retrieve relation by query and text chunk.

        Args:
            query_id: The query ID.
            chunk_id: The text chunk ID.

        Returns:
            The retrieval relation if found, None otherwise.
        """
        stmt = select(self.model_cls).where(
            self.model_cls.query_id == query_id,
            self.model_cls.chunk_id == chunk_id,
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def get_by_query_and_image_chunk(self, query_id: int, image_chunk_id: int) -> Any | None:
        """Retrieve relation by query and image chunk.

        Args:
            query_id: The query ID.
            image_chunk_id: The image chunk ID.

        Returns:
            The retrieval relation if found, None otherwise.
        """
        stmt = select(self.model_cls).where(
            self.model_cls.query_id == query_id,
            self.model_cls.image_chunk_id == image_chunk_id,
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def get_max_group_index(self, query_id: int) -> int | None:
        """Get the maximum group_index for a query.

        Args:
            query_id: The query ID.

        Returns:
            Maximum group_index value, None if no relations exist.
        """
        stmt = select(func.max(self.model_cls.group_index)).where(self.model_cls.query_id == query_id)
        return self.session.execute(stmt).scalar_one_or_none()

    def get_max_group_order(self, query_id: int, group_index: int) -> int | None:
        """Get the maximum group_order for a specific group.

        Args:
            query_id: The query ID.
            group_index: The group index.

        Returns:
            Maximum group_order value, None if no relations exist.
        """
        stmt = select(func.max(self.model_cls.group_order)).where(
            self.model_cls.query_id == query_id,
            self.model_cls.group_index == group_index,
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def count_by_query(self, query_id: int) -> int:
        """Count the number of retrieval relations for a query.

        Args:
            query_id: The query ID.

        Returns:
            Number of retrieval relations for the query.
        """
        stmt = select(func.count()).select_from(self.model_cls).where(self.model_cls.query_id == query_id)
        return self.session.execute(stmt).scalar_one()

    def count_text_chunks_by_query(self, query_id: int) -> int:
        """Count text chunk relations for a query.

        Args:
            query_id: The query ID.

        Returns:
            Number of text chunk relations for the query.
        """
        stmt = (
            select(func.count())
            .select_from(self.model_cls)
            .where(
                self.model_cls.query_id == query_id,
                self.model_cls.chunk_id.is_not(None),
            )
        )
        return self.session.execute(stmt).scalar_one()

    def count_image_chunks_by_query(self, query_id: int) -> int:
        """Count image chunk relations for a query.

        Args:
            query_id: The query ID.

        Returns:
            Number of image chunk relations for the query.
        """
        stmt = (
            select(func.count())
            .select_from(self.model_cls)
            .where(
                self.model_cls.query_id == query_id,
                self.model_cls.image_chunk_id.is_not(None),
            )
        )
        return self.session.execute(stmt).scalar_one()
