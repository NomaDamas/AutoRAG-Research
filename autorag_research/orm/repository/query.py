"""Query repository for AutoRAG-Research.

Implements query-specific CRUD operations and relationship queries
for managing evaluation queries and their ground truth data.
"""

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from autorag_research.orm.repository.base import BaseEmbeddingRepository, BaseVectorRepository


class QueryRepository(BaseVectorRepository[Any], BaseEmbeddingRepository[Any]):
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
        stmt = select(self.model_cls).where(self.model_cls.contents == query_text)
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
        stmt = select(self.model_cls).where(self.model_cls.contents.ilike(f"%{search_text}%")).limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def get_queries_with_generation_gt(self) -> list[Any]:
        """Retrieve all queries that have generation ground truth.

        Returns:
            List of queries with generation ground truth.
        """
        stmt = select(self.model_cls).where(self.model_cls.generation_gt.is_not(None))
        return list(self.session.execute(stmt).scalars().all())

    def get_queries_with_empty_content(self, limit: int = 100) -> list[Any]:
        """Retrieve queries with empty or whitespace-only content.

        Args:
            limit: Maximum number of queries to retrieve.

        Returns:
            List of queries with empty content.
        """
        stmt = select(self.model_cls).where(func.trim(self.model_cls.contents) == "").limit(limit)
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
