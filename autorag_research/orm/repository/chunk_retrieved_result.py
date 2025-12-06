"""ChunkRetrievedResult repository for AutoRAG-Research.

Implements CRUD operations for storing and querying text chunk retrieval results
from various retrieval pipelines like BM25, dense retrieval, etc.
"""

from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from autorag_research.orm.repository.base import GenericRepository


class BaseRetrievedResultRepository(GenericRepository[Any]):
    """Base repository for retrieved result entities.

    Provides common CRUD operations for chunk and image chunk retrieval results.
    Subclasses should override __init__ to provide the appropriate model class.
    """

    def get_by_query_and_pipeline(self, query_id: int, pipeline_id: int) -> list[Any]:
        """Retrieve all retrieved results for a specific query and pipeline.

        Args:
            query_id: The query ID.
            pipeline_id: The pipeline ID.

        Returns:
            List of retrieved result entities ordered by relevance score descending.
        """
        stmt = (
            select(self.model_cls)
            .where(self.model_cls.query_id == query_id, self.model_cls.pipeline_id == pipeline_id)
            .order_by(self.model_cls.rel_score.desc())
        )
        return list(self.session.execute(stmt).scalars().all())

    def get_by_pipeline(self, pipeline_id: int, limit: int | None = None) -> list[Any]:
        """Retrieve all retrieved results for a specific pipeline.

        Args:
            pipeline_id: The pipeline ID.
            limit: Maximum number of results to return.

        Returns:
            List of retrieved result entities.
        """
        stmt = select(self.model_cls).where(self.model_cls.pipeline_id == pipeline_id)
        if limit:
            stmt = stmt.limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def get_by_query(self, query_id: int) -> list[Any]:
        """Retrieve all retrieved results for a specific query.

        Args:
            query_id: The query ID.

        Returns:
            List of retrieved result entities.
        """
        stmt = select(self.model_cls).where(self.model_cls.query_id == query_id)
        return list(self.session.execute(stmt).scalars().all())

    def delete_by_pipeline(self, pipeline_id: int) -> int:
        """Delete all retrieved results for a specific pipeline.

        Args:
            pipeline_id: The pipeline ID.

        Returns:
            Number of deleted records.
        """
        stmt = delete(self.model_cls).where(self.model_cls.pipeline_id == pipeline_id)
        result = self.session.execute(stmt)
        return result.rowcount  # ty: ignore

    def delete_by_query_and_pipeline(self, query_id: int, pipeline_id: int) -> int:
        """Delete all retrieved results for a specific query and pipeline.

        Args:
            query_id: The query ID.
            pipeline_id: The pipeline ID.

        Returns:
            Number of deleted records.
        """
        stmt = delete(self.model_cls).where(
            self.model_cls.query_id == query_id, self.model_cls.pipeline_id == pipeline_id
        )
        result = self.session.execute(stmt)
        return result.rowcount  # ty: ignore

    def delete_by_composite_key(self, query_id: int, pipeline_id: int, chunk_id: int) -> int:
        """Delete a retrieved result by its composite key.

        Args:
            query_id: The query ID.
            pipeline_id: The pipeline ID.
            chunk_id: The chunk ID.

        Returns:
            Number of deleted records.
        """
        stmt = delete(self.model_cls).where(
            self.model_cls.query_id == query_id,
            self.model_cls.pipeline_id == pipeline_id,
            self.model_cls.chunk_id == chunk_id,
        )
        result = self.session.execute(stmt)
        return result.rowcount  # ty: ignore

    def bulk_insert(self, results: list[dict]) -> int:
        """Bulk insert retrieved results.

        Args:
            results: List of dictionaries containing result data.

        Returns:
            Number of inserted records.
        """
        entities = [self.model_cls(**r) for r in results]
        self.session.add_all(entities)
        return len(entities)


class ChunkRetrievedResultRepository(BaseRetrievedResultRepository):
    """Repository for ChunkRetrievedResult entity."""

    def __init__(self, session: Session, model_cls: type | None = None):
        """Initialize chunk retrieved result repository.

        Args:
            session: SQLAlchemy session for database operations.
            model_cls: The ChunkRetrievedResult model class to use. If None, uses default schema.
        """
        if model_cls is None:
            from autorag_research.orm.schema import ChunkRetrievedResult

            model_cls = ChunkRetrievedResult
        super().__init__(session, model_cls)
