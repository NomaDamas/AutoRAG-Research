"""Query repository for AutoRAG-Research.

Implements query-specific CRUD operations and relationship queries
for managing evaluation queries and their ground truth data.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from autorag_research.orm.repository.base import BaseVectorRepository
from autorag_research.orm.schema import Query


class QueryRepository(BaseVectorRepository[Query]):
    """Repository for Query entity with relationship loading and vector search capabilities."""

    def __init__(self, session: Session):
        """Initialize query repository.

        Args:
            session: SQLAlchemy session for database operations.
        """
        super().__init__(session, Query)

    def get_by_query_text(self, query_text: str) -> Query | None:
        """Retrieve a query by its text content.

        Args:
            query_text: The query text to search for.

        Returns:
            The query if found, None otherwise.
        """
        stmt = select(Query).where(Query.query == query_text)
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def get_with_retrieval_relations(self, query_id: int) -> Query | None:
        """Retrieve a query with its retrieval relations eagerly loaded.

        Args:
            query_id: The query ID.

        Returns:
            The query with retrieval relations loaded, None if not found.
        """
        stmt = select(Query).where(Query.id == query_id).options(joinedload(Query.retrieval_relations))
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def get_with_executor_results(self, query_id: int) -> Query | None:
        """Retrieve a query with its executor results eagerly loaded.

        Args:
            query_id: The query ID.

        Returns:
            The query with executor results loaded, None if not found.
        """
        stmt = select(Query).where(Query.id == query_id).options(joinedload(Query.executor_results))
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def get_with_all_relations(self, query_id: int) -> Query | None:
        """Retrieve a query with all relations eagerly loaded.

        Args:
            query_id: The query ID.

        Returns:
            The query with all relations loaded, None if not found.
        """
        stmt = (
            select(Query)
            .where(Query.id == query_id)
            .options(
                joinedload(Query.retrieval_relations),
                joinedload(Query.executor_results),
                joinedload(Query.chunk_retrieved_results),
                joinedload(Query.image_chunk_retrieved_results),
                joinedload(Query.evaluation_results),
            )
        )
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def search_by_query_text(self, search_text: str, limit: int = 10) -> list[Query]:
        """Search queries containing the specified text.

        Args:
            search_text: Text to search for in query content.
            limit: Maximum number of results to return.

        Returns:
            List of queries containing the search text.
        """
        stmt = select(Query).where(Query.query.ilike(f"%{search_text}%")).limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def get_queries_with_generation_gt(self) -> list[Query]:
        """Retrieve all queries that have generation ground truth.

        Returns:
            List of queries with generation ground truth.
        """
        stmt = select(Query).where(Query.generation_gt.is_not(None))
        return list(self.session.execute(stmt).scalars().all())

    def count_by_generation_gt_size(self, size: int) -> int:
        """Count queries with a specific number of generation ground truths.

        Args:
            size: The number of ground truths to match.

        Returns:
            Count of queries with the specified number of ground truths.
        """
        from sqlalchemy import func

        stmt = select(func.count()).select_from(Query).where(func.array_length(Query.generation_gt, 1) == size)
        return self.session.execute(stmt).scalar_one()

    def get_queries_without_embeddings(self, limit: int | None = None, offset: int | None = None) -> list[Query]:
        """Retrieve queries that do not have embeddings.

        Args:
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of queries without embeddings.
        """
        stmt = select(Query).where(Query.embedding.is_(None))
        if offset:
            stmt = stmt.offset(offset)
        if limit:
            stmt = stmt.limit(limit)
        return list(self.session.execute(stmt).scalars().all())
