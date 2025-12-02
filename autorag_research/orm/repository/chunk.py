"""Chunk repository for AutoRAG-Research.

Implements chunk-specific CRUD operations and queries extending
the base vector repository pattern for similarity search.
"""

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from autorag_research.orm.repository.base import BaseEmbeddingRepository, BaseVectorRepository


class ChunkRepository(BaseVectorRepository[Any], BaseEmbeddingRepository[Any]):
    """Repository for Chunk entity with vector search capabilities."""

    def __init__(self, session: Session, model_cls: type | None = None):
        """Initialize chunk repository.

        Args:
            session: SQLAlchemy session for database operations.
            model_cls: The Chunk model class to use. If None, uses default schema.
        """
        if model_cls is None:
            from autorag_research.orm.schema import Chunk

            model_cls = Chunk
        super().__init__(session, model_cls)

    def get_by_caption_id(self, caption_id: int) -> list[Any]:
        """Retrieve all chunks for a specific caption.

        Args:
            caption_id: The caption ID.

        Returns:
            List of chunks belonging to the caption.
        """
        stmt = select(self.model_cls).where(self.model_cls.parent_caption == caption_id)
        return list(self.session.execute(stmt).scalars().all())

    def get_with_parent_caption(self, chunk_id: int) -> Any | None:
        """Retrieve a chunk with its parent caption eagerly loaded.

        Args:
            chunk_id: The chunk ID.

        Returns:
            The chunk with parent caption loaded, None if not found.
        """
        stmt = (
            select(self.model_cls)
            .where(self.model_cls.id == chunk_id)
            .options(joinedload(self.model_cls.parent_caption_obj))
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def get_with_caption_chunk_relations(self, chunk_id: int) -> Any | None:
        """Retrieve a chunk with its caption-chunk relations eagerly loaded.

        Args:
            chunk_id: The chunk ID.

        Returns:
            The chunk with caption-chunk relations loaded, None if not found.
        """
        stmt = (
            select(self.model_cls)
            .where(self.model_cls.id == chunk_id)
            .options(joinedload(self.model_cls.caption_chunk_relations))
        )
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def get_with_retrieval_relations(self, chunk_id: int) -> Any | None:
        """Retrieve a chunk with its retrieval relations eagerly loaded.

        Args:
            chunk_id: The chunk ID.

        Returns:
            The chunk with retrieval relations loaded, None if not found.
        """
        stmt = (
            select(self.model_cls)
            .where(self.model_cls.id == chunk_id)
            .options(joinedload(self.model_cls.retrieval_relations))
        )
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def get_with_chunk_retrieved_results(self, chunk_id: int) -> Any | None:
        """Retrieve a chunk with its chunk retrieved results eagerly loaded.

        Args:
            chunk_id: The chunk ID.

        Returns:
            The chunk with chunk retrieved results loaded, None if not found.
        """
        stmt = (
            select(self.model_cls)
            .where(self.model_cls.id == chunk_id)
            .options(joinedload(self.model_cls.chunk_retrieved_results))
        )
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def search_by_contents(self, search_text: str) -> list[Any]:
        """Search chunks by contents using SQL LIKE.

        Args:
            search_text: The text to search for (use % as wildcard).

        Returns:
            List of matching chunks.
        """
        stmt = select(self.model_cls).where(self.model_cls.contents.like(f"%{search_text}%"))
        return list(self.session.execute(stmt).scalars().all())

    def get_by_contents_exact(self, contents: str) -> list[Any]:
        """Retrieve chunks with exact contents match.

        Args:
            contents: The exact contents to search for.

        Returns:
            List of chunks with matching contents.
        """
        stmt = select(self.model_cls).where(self.model_cls.contents == contents)
        return list(self.session.execute(stmt).scalars().all())

    def count_by_caption(self, caption_id: int) -> int:
        """Count the number of chunks for a specific caption.

        Args:
            caption_id: The caption ID.

        Returns:
            Number of chunks for the caption.
        """
        return self.session.query(self.model_cls).filter(self.model_cls.parent_caption == caption_id).count()

    def get_with_all_relations(self, chunk_id: int) -> Any | None:
        """Retrieve a chunk with all relationships eagerly loaded.

        Args:
            chunk_id: The chunk ID.

        Returns:
            The chunk with all relations loaded, None if not found.
        """
        stmt = (
            select(self.model_cls)
            .where(self.model_cls.id == chunk_id)
            .options(
                joinedload(self.model_cls.parent_caption_obj),
                joinedload(self.model_cls.caption_chunk_relations),
                joinedload(self.model_cls.retrieval_relations),
                joinedload(self.model_cls.chunk_retrieved_results),
            )
        )
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def get_chunks_with_empty_content(self, limit: int | None = None) -> list[Any]:
        """Retrieve chunks that have empty or whitespace-only contents.

        Args:
            limit: Maximum number of results to return.

        Returns:
            List of chunks with empty content.
        """
        # Use SQL TRIM to check for empty or whitespace-only content
        stmt = select(self.model_cls).where(
            (self.model_cls.contents.is_(None)) | (func.trim(self.model_cls.contents) == "")
        )
        if limit:
            stmt = stmt.limit(limit)
        return list(self.session.execute(stmt).scalars().all())
