"""Query repository for AutoRAG-Research.

Implements query-specific CRUD operations and relationship queries
for managing evaluation queries and their ground truth data.
"""

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from autorag_research.orm.repository.base import BaseVectorRepository


class QueryRepository(BaseVectorRepository[Any]):
    """Repository for Query entity with relationship loading and vector search capabilities."""

    def __init__(self, session: Session, model_cls: type | None = None):
        """Initialize query repository.

        Args:
            session: SQLAlchemy session for database operations.
            model_cls: The Query model class to use. If None, uses default schema.
        """
        if model_cls is None:
            from autorag_research.orm.schema import Query

            model_cls = Query
        super().__init__(session, model_cls)

    def get_by_query_text(self, query_text: str) -> Any | None:
        """Retrieve a query by its text content.

        Args:
            query_text: The query text to search for.

        Returns:
            The query if found, None otherwise.
        """
        stmt = select(self.model_cls).where(self.model_cls.query == query_text)
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def get_with_retrieval_relations(self, query_id: int) -> Any | None:
        """Retrieve a query with its retrieval relations eagerly loaded.

        Args:
            query_id: The query ID.

        Returns:
            The query with retrieval relations loaded, None if not found.
        """
        stmt = (
            select(self.model_cls)
            .where(self.model_cls.id == query_id)
            .options(joinedload(self.model_cls.retrieval_relations))
        )
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def get_with_executor_results(self, query_id: int) -> Any | None:
        """Retrieve a query with its executor results eagerly loaded.

        Args:
            query_id: The query ID.

        Returns:
            The query with executor results loaded, None if not found.
        """
        stmt = (
            select(self.model_cls)
            .where(self.model_cls.id == query_id)
            .options(joinedload(self.model_cls.executor_results))
        )
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def get_with_all_relations(self, query_id: int) -> Any | None:
        """Retrieve a query with all relations eagerly loaded.

        Args:
            query_id: The query ID.

        Returns:
            The query with all relations loaded, None if not found.
        """
        stmt = (
            select(self.model_cls)
            .where(self.model_cls.id == query_id)
            .options(
                joinedload(self.model_cls.retrieval_relations),
                joinedload(self.model_cls.executor_results),
                joinedload(self.model_cls.chunk_retrieved_results),
                joinedload(self.model_cls.image_chunk_retrieved_results),
                joinedload(self.model_cls.evaluation_results),
            )
        )
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def search_by_query_text(self, search_text: str, limit: int = 10) -> list[Any]:
        """Search queries containing the specified text.

        Args:
            search_text: Text to search for in query content.
            limit: Maximum number of results to return.

        Returns:
            List of queries containing the search text.
        """
        stmt = select(self.model_cls).where(self.model_cls.query.ilike(f"%{search_text}%")).limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def get_queries_with_generation_gt(self) -> list[Any]:
        """Retrieve all queries that have generation ground truth.

        Returns:
            List of queries with generation ground truth.
        """
        stmt = select(self.model_cls).where(self.model_cls.generation_gt.is_not(None))
        return list(self.session.execute(stmt).scalars().all())

    def count_by_generation_gt_size(self, size: int) -> int:
        """Count queries with a specific number of generation ground truths.

        Args:
            size: The number of ground truths to match.

        Returns:
            Count of queries with the specified number of ground truths.
        """
        stmt = (
            select(func.count())
            .select_from(self.model_cls)
            .where(func.array_length(self.model_cls.generation_gt, 1) == size)
        )
        return self.session.execute(stmt).scalar_one()

    def get_queries_without_embeddings(self, limit: int | None = None, offset: int | None = None) -> list[Any]:
        """Retrieve queries that do not have embeddings.

        Args:
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of queries without embeddings.
        """
        stmt = select(self.model_cls).where(self.model_cls.embedding.is_(None))
        if offset:
            stmt = stmt.offset(offset)
        if limit:
            stmt = stmt.limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def get_queries_with_empty_content(self, limit: int | None = None) -> list[Any]:
        """Retrieve queries that have empty or whitespace-only query text.

        Args:
            limit: Maximum number of results to return.

        Returns:
            List of queries with empty content.
        """
        # Use SQL TRIM to check for empty or whitespace-only content
        stmt = select(self.model_cls).where((self.model_cls.query.is_(None)) | (func.trim(self.model_cls.query) == ""))
        if limit:
            stmt = stmt.limit(limit)
        return list(self.session.execute(stmt).scalars().all())
