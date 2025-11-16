"""Chunk repository for AutoRAG-Research.

Implements chunk-specific CRUD operations and queries extending
the base vector repository pattern for similarity search.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from autorag_research.orm.repository.base import BaseVectorRepository
from autorag_research.orm.schema import Chunk


class ChunkRepository(BaseVectorRepository[Chunk]):
    """Repository for Chunk entity with vector search capabilities."""

    def __init__(self, session: Session):
        """Initialize chunk repository.

        Args:
            session: SQLAlchemy session for database operations.
        """
        super().__init__(session, Chunk)

    def get_by_caption_id(self, caption_id: int) -> list[Chunk]:
        """Retrieve all chunks for a specific caption.

        Args:
            caption_id: The caption ID.

        Returns:
            List of chunks belonging to the caption.
        """
        stmt = select(Chunk).where(Chunk.parent_caption == caption_id)
        return list(self.session.execute(stmt).scalars().all())

    def get_with_parent_caption(self, chunk_id: int) -> Chunk | None:
        """Retrieve a chunk with its parent caption eagerly loaded.

        Args:
            chunk_id: The chunk ID.

        Returns:
            The chunk with parent caption loaded, None if not found.
        """
        stmt = select(Chunk).where(Chunk.id == chunk_id).options(joinedload(Chunk.parent_caption_obj))
        return self.session.execute(stmt).scalar_one_or_none()

    def get_with_caption_chunk_relations(self, chunk_id: int) -> Chunk | None:
        """Retrieve a chunk with its caption-chunk relations eagerly loaded.

        Args:
            chunk_id: The chunk ID.

        Returns:
            The chunk with caption-chunk relations loaded, None if not found.
        """
        stmt = select(Chunk).where(Chunk.id == chunk_id).options(joinedload(Chunk.caption_chunk_relations))
        return self.session.execute(stmt).scalar_one_or_none()

    def get_with_retrieval_relations(self, chunk_id: int) -> Chunk | None:
        """Retrieve a chunk with its retrieval relations eagerly loaded.

        Args:
            chunk_id: The chunk ID.

        Returns:
            The chunk with retrieval relations loaded, None if not found.
        """
        stmt = select(Chunk).where(Chunk.id == chunk_id).options(joinedload(Chunk.retrieval_relations))
        return self.session.execute(stmt).scalar_one_or_none()

    def get_with_chunk_retrieved_results(self, chunk_id: int) -> Chunk | None:
        """Retrieve a chunk with its chunk retrieved results eagerly loaded.

        Args:
            chunk_id: The chunk ID.

        Returns:
            The chunk with chunk retrieved results loaded, None if not found.
        """
        stmt = select(Chunk).where(Chunk.id == chunk_id).options(joinedload(Chunk.chunk_retrieved_results))
        return self.session.execute(stmt).scalar_one_or_none()

    def search_by_contents(self, search_text: str) -> list[Chunk]:
        """Search chunks by contents using SQL LIKE.

        Args:
            search_text: The text to search for (use % as wildcard).

        Returns:
            List of matching chunks.
        """
        stmt = select(Chunk).where(Chunk.contents.like(f"%{search_text}%"))
        return list(self.session.execute(stmt).scalars().all())

    def get_by_contents_exact(self, contents: str) -> list[Chunk]:
        """Retrieve chunks with exact contents match.

        Args:
            contents: The exact contents to search for.

        Returns:
            List of chunks with matching contents.
        """
        stmt = select(Chunk).where(Chunk.contents == contents)
        return list(self.session.execute(stmt).scalars().all())

    def get_chunks_with_embeddings(self, limit: int | None = None, offset: int | None = None) -> list[Chunk]:
        """Retrieve chunks that have embeddings.

        Args:
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of chunks with embeddings.
        """
        stmt = select(Chunk).where(Chunk.embedding.is_not(None))
        if offset:
            stmt = stmt.offset(offset)
        if limit:
            stmt = stmt.limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def get_chunks_without_embeddings(self, limit: int | None = None, offset: int | None = None) -> list[Chunk]:
        """Retrieve chunks that do not have embeddings.

        Args:
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of chunks without embeddings.
        """
        stmt = select(Chunk).where(Chunk.embedding.is_(None))
        if offset:
            stmt = stmt.offset(offset)
        if limit:
            stmt = stmt.limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def count_by_caption(self, caption_id: int) -> int:
        """Count the number of chunks for a specific caption.

        Args:
            caption_id: The caption ID.

        Returns:
            Number of chunks for the caption.
        """
        return self.session.query(Chunk).filter(Chunk.parent_caption == caption_id).count()

    def get_with_all_relations(self, chunk_id: int) -> Chunk | None:
        """Retrieve a chunk with all relationships eagerly loaded.

        Args:
            chunk_id: The chunk ID.

        Returns:
            The chunk with all relations loaded, None if not found.
        """
        stmt = (
            select(Chunk)
            .where(Chunk.id == chunk_id)
            .options(
                joinedload(Chunk.parent_caption_obj),
                joinedload(Chunk.caption_chunk_relations),
                joinedload(Chunk.retrieval_relations),
                joinedload(Chunk.chunk_retrieved_results),
            )
        )
        return self.session.execute(stmt).scalar_one_or_none()
