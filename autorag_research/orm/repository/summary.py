"""Summary repository for AutoRAG-Research.

Implements summary-specific CRUD operations and relationship queries
for managing aggregated pipeline metrics and results.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from autorag_research.orm.repository.base import GenericRepository
from autorag_research.orm.schema import Summary


class SummaryRepository(GenericRepository[Summary]):
    """Repository for Summary entity with composite key support."""

    def __init__(self, session: Session):
        """Initialize summary repository.

        Args:
            session: SQLAlchemy session for database operations.
        """
        super().__init__(session, Summary)

    def get_by_composite_key(self, pipeline_id: int, metric_id: int) -> Summary | None:
        """Retrieve a summary by its composite primary key.

        Args:
            pipeline_id: The pipeline ID.
            metric_id: The metric ID.

        Returns:
            The summary if found, None otherwise.
        """
        stmt = select(Summary).where(
            Summary.pipeline_id == pipeline_id,
            Summary.metric_id == metric_id,
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def get_by_pipeline_id(self, pipeline_id: int) -> list[Summary]:
        """Retrieve all summaries for a specific pipeline.

        Args:
            pipeline_id: The pipeline ID.

        Returns:
            List of summaries for the pipeline.
        """
        stmt = select(Summary).where(Summary.pipeline_id == pipeline_id)
        return list(self.session.execute(stmt).scalars().all())

    def get_by_metric_id(self, metric_id: int) -> list[Summary]:
        """Retrieve all summaries for a specific metric.

        Args:
            metric_id: The metric ID.

        Returns:
            List of summaries for the metric.
        """
        stmt = select(Summary).where(Summary.metric_id == metric_id)
        return list(self.session.execute(stmt).scalars().all())

    def get_with_all_relations(self, pipeline_id: int, metric_id: int) -> Summary | None:
        """Retrieve a summary with all relations eagerly loaded.

        Args:
            pipeline_id: The pipeline ID.
            metric_id: The metric ID.

        Returns:
            The summary with all relations loaded, None if not found.
        """
        stmt = (
            select(Summary)
            .where(
                Summary.pipeline_id == pipeline_id,
                Summary.metric_id == metric_id,
            )
            .options(
                joinedload(Summary.pipeline),
                joinedload(Summary.metric),
            )
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def get_by_metric_result_range(self, metric_id: int, min_result: float, max_result: float) -> list[Summary]:
        """Retrieve summaries within a metric result range.

        Args:
            metric_id: The metric ID.
            min_result: Minimum metric result value (inclusive).
            max_result: Maximum metric result value (inclusive).

        Returns:
            List of summaries within the specified range.
        """
        stmt = select(Summary).where(
            Summary.metric_id == metric_id,
            Summary.metric_result >= min_result,
            Summary.metric_result <= max_result,
        )
        return list(self.session.execute(stmt).scalars().all())

    def get_top_pipelines_by_metric(self, metric_id: int, limit: int = 10, ascending: bool = False) -> list[Summary]:
        """Retrieve top performing pipelines for a specific metric.

        Args:
            metric_id: The metric ID.
            limit: Maximum number of results to return.
            ascending: If True, sort ascending (lower is better), otherwise descending (higher is better).

        Returns:
            List of summaries ordered by metric result.
        """
        stmt = select(Summary).where(Summary.metric_id == metric_id)
        stmt = stmt.order_by(Summary.metric_result.asc()) if ascending else stmt.order_by(Summary.metric_result.desc())
        stmt = stmt.limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def get_pipeline_summaries_with_relations(self, pipeline_id: int) -> list[Summary]:
        """Retrieve all summaries for a pipeline with relations eagerly loaded.

        Args:
            pipeline_id: The pipeline ID.

        Returns:
            List of summaries with pipeline and metric loaded.
        """
        stmt = (
            select(Summary)
            .where(Summary.pipeline_id == pipeline_id)
            .options(
                joinedload(Summary.pipeline),
                joinedload(Summary.metric),
            )
        )
        return list(self.session.execute(stmt).scalars().all())

    def get_metric_summaries_with_relations(self, metric_id: int) -> list[Summary]:
        """Retrieve all summaries for a metric with relations eagerly loaded.

        Args:
            metric_id: The metric ID.

        Returns:
            List of summaries with pipeline and metric loaded.
        """
        stmt = (
            select(Summary)
            .where(Summary.metric_id == metric_id)
            .options(
                joinedload(Summary.pipeline),
                joinedload(Summary.metric),
            )
        )
        return list(self.session.execute(stmt).scalars().all())

    def compare_pipelines_by_metric(self, pipeline_ids: list[int], metric_id: int) -> list[Summary]:
        """Compare multiple pipelines on a specific metric.

        Args:
            pipeline_ids: List of pipeline IDs to compare.
            metric_id: The metric ID to compare on.

        Returns:
            List of summaries for the specified pipelines and metric, ordered by metric result.
        """
        stmt = (
            select(Summary)
            .where(
                Summary.pipeline_id.in_(pipeline_ids),
                Summary.metric_id == metric_id,
            )
            .order_by(Summary.metric_result.desc())
        )
        return list(self.session.execute(stmt).scalars().all())

    def delete_by_composite_key(self, pipeline_id: int, metric_id: int) -> bool:
        """Delete a summary by its composite primary key.

        Args:
            pipeline_id: The pipeline ID.
            metric_id: The metric ID.

        Returns:
            True if the summary was deleted, False if not found.
        """
        summary = self.get_by_composite_key(pipeline_id, metric_id)
        if summary:
            self.delete(summary)
            return True
        return False

    def exists_by_composite_key(self, pipeline_id: int, metric_id: int) -> bool:
        """Check if a summary exists with the given composite key.

        Args:
            pipeline_id: The pipeline ID.
            metric_id: The metric ID.

        Returns:
            True if the summary exists, False otherwise.
        """
        return self.get_by_composite_key(pipeline_id, metric_id) is not None

    def get_by_execution_time_range(self, pipeline_id: int, min_time: int, max_time: int) -> list[Summary]:
        """Retrieve summaries within an execution time range.

        Args:
            pipeline_id: The pipeline ID.
            min_time: Minimum execution time (inclusive).
            max_time: Maximum execution time (inclusive).

        Returns:
            List of summaries within the specified time range.
        """
        stmt = select(Summary).where(
            Summary.pipeline_id == pipeline_id,
            Summary.execution_time >= min_time,
            Summary.execution_time <= max_time,
        )
        return list(self.session.execute(stmt).scalars().all())
