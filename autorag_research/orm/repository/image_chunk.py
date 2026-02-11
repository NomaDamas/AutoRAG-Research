"""ImageChunk repository for AutoRAG-Research.

Implements image chunk-specific CRUD operations and queries extending
the base vector repository pattern for similarity search.
"""

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from autorag_research.orm.repository.base import BaseEmbeddingRepository, BaseVectorRepository


class ImageChunkRepository(BaseVectorRepository[Any], BaseEmbeddingRepository[Any]):
    """Repository for ImageChunk entity with vector search capabilities."""

    def __init__(self, session: Session, model_cls: type | None = None):
        """Initialize image chunk repository.

        Args:
            session: SQLAlchemy session for database operations.
            model_cls: The ImageChunk model class to use. If None, uses default schema.
        """
        if model_cls is None:
            from autorag_research.orm.schema import ImageChunk

            model_cls = ImageChunk
        super().__init__(session, model_cls)

    def get_by_page_id(self, page_id: int | str) -> list[Any]:
        """Retrieve all image chunks for a specific page.

        Args:
            page_id: The page ID.

        Returns:
            List of image chunks belonging to the page.
        """
        stmt = select(self.model_cls).where(self.model_cls.parent_page == page_id)
        return list(self.session.execute(stmt).scalars().all())

    def get_with_page(self, image_chunk_id: int | str) -> Any | None:
        """Retrieve an image chunk with its page eagerly loaded.

        Args:
            image_chunk_id: The image chunk ID.

        Returns:
            The image chunk with page loaded, None if not found.
        """
        stmt = (
            select(self.model_cls).where(self.model_cls.id == image_chunk_id).options(joinedload(self.model_cls.page))
        )
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def get_with_retrieval_relations(self, image_chunk_id: int | str) -> Any | None:
        """Retrieve an image chunk with its retrieval relations eagerly loaded.

        Args:
            image_chunk_id: The image chunk ID.

        Returns:
            The image chunk with retrieval relations loaded, None if not found.
        """
        stmt = (
            select(self.model_cls)
            .where(self.model_cls.id == image_chunk_id)
            .options(joinedload(self.model_cls.retrieval_relations))
        )
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def get_with_image_chunk_retrieved_results(self, image_chunk_id: int | str) -> Any | None:
        """Retrieve an image chunk with its image chunk retrieved results eagerly loaded.

        Args:
            image_chunk_id: The image chunk ID.

        Returns:
            The image chunk with image chunk retrieved results loaded, None if not found.
        """
        stmt = (
            select(self.model_cls)
            .where(self.model_cls.id == image_chunk_id)
            .options(joinedload(self.model_cls.image_chunk_retrieved_results))
        )
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def count_by_page(self, page_id: int | str) -> int:
        """Count the number of image chunks for a specific page.

        Args:
            page_id: The page ID.

        Returns:
            Number of image chunks for the page.
        """
        return self.session.query(self.model_cls).filter(self.model_cls.parent_page == page_id).count()

    def get_with_all_relations(self, image_chunk_id: int | str) -> Any | None:
        """Retrieve an image chunk with all relationships eagerly loaded.

        Args:
            image_chunk_id: The image chunk ID.

        Returns:
            The image chunk with all relations loaded, None if not found.
        """
        stmt = (
            select(self.model_cls)
            .where(self.model_cls.id == image_chunk_id)
            .options(
                joinedload(self.model_cls.page),
                joinedload(self.model_cls.retrieval_relations),
                joinedload(self.model_cls.image_chunk_retrieved_results),
            )
        )
        return self.session.execute(stmt).unique().scalar_one_or_none()
