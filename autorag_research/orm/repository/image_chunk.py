"""ImageChunk repository for AutoRAG-Research.

Implements image chunk-specific CRUD operations and queries extending
the base vector repository pattern for similarity search.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from autorag_research.orm.repository.base import BaseVectorRepository
from autorag_research.orm.schema import ImageChunk


class ImageChunkRepository(BaseVectorRepository[ImageChunk]):
    """Repository for ImageChunk entity with vector search capabilities."""

    def __init__(self, session: Session):
        """Initialize image chunk repository.

        Args:
            session: SQLAlchemy session for database operations.
        """
        super().__init__(session, ImageChunk)

    def get_by_page_id(self, page_id: int) -> list[ImageChunk]:
        """Retrieve all image chunks for a specific page.

        Args:
            page_id: The page ID.

        Returns:
            List of image chunks belonging to the page.
        """
        stmt = select(ImageChunk).where(ImageChunk.parent_page == page_id)
        return list(self.session.execute(stmt).scalars().all())

    def get_by_image_path_id(self, image_path_id: int) -> ImageChunk | None:
        """Retrieve an image chunk by its image path ID.

        Args:
            image_path_id: The file ID for the image.

        Returns:
            The image chunk if found, None otherwise.
        """
        stmt = select(ImageChunk).where(ImageChunk.image_path == image_path_id)
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def get_with_page(self, image_chunk_id: int) -> ImageChunk | None:
        """Retrieve an image chunk with its page eagerly loaded.

        Args:
            image_chunk_id: The image chunk ID.

        Returns:
            The image chunk with page loaded, None if not found.
        """
        stmt = select(ImageChunk).where(ImageChunk.id == image_chunk_id).options(joinedload(ImageChunk.page))
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def get_with_image_file(self, image_chunk_id: int) -> ImageChunk | None:
        """Retrieve an image chunk with its image file eagerly loaded.

        Args:
            image_chunk_id: The image chunk ID.

        Returns:
            The image chunk with image file loaded, None if not found.
        """
        stmt = select(ImageChunk).where(ImageChunk.id == image_chunk_id).options(joinedload(ImageChunk.image_file))
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def get_with_retrieval_relations(self, image_chunk_id: int) -> ImageChunk | None:
        """Retrieve an image chunk with its retrieval relations eagerly loaded.

        Args:
            image_chunk_id: The image chunk ID.

        Returns:
            The image chunk with retrieval relations loaded, None if not found.
        """
        stmt = (
            select(ImageChunk)
            .where(ImageChunk.id == image_chunk_id)
            .options(joinedload(ImageChunk.retrieval_relations))
        )
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def get_with_image_chunk_retrieved_results(self, image_chunk_id: int) -> ImageChunk | None:
        """Retrieve an image chunk with its image chunk retrieved results eagerly loaded.

        Args:
            image_chunk_id: The image chunk ID.

        Returns:
            The image chunk with image chunk retrieved results loaded, None if not found.
        """
        stmt = (
            select(ImageChunk)
            .where(ImageChunk.id == image_chunk_id)
            .options(joinedload(ImageChunk.image_chunk_retrieved_results))
        )
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def get_image_chunks_with_embeddings(self, limit: int | None = None, offset: int | None = None) -> list[ImageChunk]:
        """Retrieve image chunks that have embeddings.

        Args:
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of image chunks with embeddings.
        """
        stmt = select(ImageChunk).where(ImageChunk.embedding.is_not(None))
        if offset:
            stmt = stmt.offset(offset)
        if limit:
            stmt = stmt.limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def get_image_chunks_without_embeddings(
        self, limit: int | None = None, offset: int | None = None
    ) -> list[ImageChunk]:
        """Retrieve image chunks that do not have embeddings.

        Args:
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of image chunks without embeddings.
        """
        stmt = select(ImageChunk).where(ImageChunk.embedding.is_(None))
        if offset:
            stmt = stmt.offset(offset)
        if limit:
            stmt = stmt.limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def count_by_page(self, page_id: int) -> int:
        """Count the number of image chunks for a specific page.

        Args:
            page_id: The page ID.

        Returns:
            Number of image chunks for the page.
        """
        return self.session.query(ImageChunk).filter(ImageChunk.parent_page == page_id).count()

    def get_with_all_relations(self, image_chunk_id: int) -> ImageChunk | None:
        """Retrieve an image chunk with all relationships eagerly loaded.

        Args:
            image_chunk_id: The image chunk ID.

        Returns:
            The image chunk with all relations loaded, None if not found.
        """
        stmt = (
            select(ImageChunk)
            .where(ImageChunk.id == image_chunk_id)
            .options(
                joinedload(ImageChunk.page),
                joinedload(ImageChunk.image_file),
                joinedload(ImageChunk.retrieval_relations),
                joinedload(ImageChunk.image_chunk_retrieved_results),
            )
        )
        return self.session.execute(stmt).unique().scalar_one_or_none()
