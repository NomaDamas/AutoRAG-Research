"""ExecutorResult repository for AutoRAG-Research.

Implements executor result-specific CRUD operations and relationship queries
for managing query-pipeline execution results.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from autorag_research.orm.repository.base import GenericRepository
from autorag_research.orm.schema import ExecutorResult


class ExecutorResultRepository(GenericRepository[ExecutorResult]):
    """Repository for ExecutorResult entity with composite key support."""

    def __init__(self, session: Session):
        """Initialize executor result repository.

        Args:
            session: SQLAlchemy session for database operations.
        """
        super().__init__(session, ExecutorResult)

    def get_by_composite_key(self, query_id: int, pipeline_id: int) -> ExecutorResult | None:
        """Retrieve an executor result by its composite primary key.

        Args:
            query_id: The query ID.
            pipeline_id: The pipeline ID.

        Returns:
            The executor result if found, None otherwise.
        """
        stmt = select(ExecutorResult).where(
            ExecutorResult.query_id == query_id,
            ExecutorResult.pipeline_id == pipeline_id,
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def get_by_query_id(self, query_id: int) -> list[ExecutorResult]:
        """Retrieve all executor results for a specific query.

        Args:
            query_id: The query ID.

        Returns:
            List of executor results for the query.
        """
        stmt = select(ExecutorResult).where(ExecutorResult.query_id == query_id)
        return list(self.session.execute(stmt).scalars().all())

    def get_by_pipeline_id(self, pipeline_id: int) -> list[ExecutorResult]:
        """Retrieve all executor results for a specific pipeline.

        Args:
            pipeline_id: The pipeline ID.

        Returns:
            List of executor results for the pipeline.
        """
        stmt = select(ExecutorResult).where(ExecutorResult.pipeline_id == pipeline_id)
        return list(self.session.execute(stmt).scalars().all())

    def get_with_all_relations(self, query_id: int, pipeline_id: int) -> ExecutorResult | None:
        """Retrieve an executor result with all relations eagerly loaded.

        Args:
            query_id: The query ID.
            pipeline_id: The pipeline ID.

        Returns:
            The executor result with all relations loaded, None if not found.
        """
        stmt = (
            select(ExecutorResult)
            .where(
                ExecutorResult.query_id == query_id,
                ExecutorResult.pipeline_id == pipeline_id,
            )
            .options(
                joinedload(ExecutorResult.query_obj),
                joinedload(ExecutorResult.pipeline),
            )
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def get_by_execution_time_range(self, pipeline_id: int, min_time: int, max_time: int) -> list[ExecutorResult]:
        """Retrieve executor results within an execution time range.

        Args:
            pipeline_id: The pipeline ID.
            min_time: Minimum execution time (inclusive).
            max_time: Maximum execution time (inclusive).

        Returns:
            List of executor results within the specified range.
        """
        stmt = select(ExecutorResult).where(
            ExecutorResult.pipeline_id == pipeline_id,
            ExecutorResult.execution_time >= min_time,
            ExecutorResult.execution_time <= max_time,
        )
        return list(self.session.execute(stmt).scalars().all())

    def delete_by_composite_key(self, query_id: int, pipeline_id: int) -> bool:
        """Delete an executor result by its composite primary key.

        Args:
            query_id: The query ID.
            pipeline_id: The pipeline ID.

        Returns:
            True if the result was deleted, False if not found.
        """
        result = self.get_by_composite_key(query_id, pipeline_id)
        if result:
            self.delete(result)
            return True
        return False

    def exists_by_composite_key(self, query_id: int, pipeline_id: int) -> bool:
        """Check if an executor result exists with the given composite key.

        Args:
            query_id: The query ID.
            pipeline_id: The pipeline ID.

        Returns:
            True if the result exists, False otherwise.
        """
        return self.get_by_composite_key(query_id, pipeline_id) is not None
