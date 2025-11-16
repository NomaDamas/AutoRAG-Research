"""ExperimentResult repository for AutoRAG-Research.

Implements experiment result-specific CRUD operations and relationship queries
for managing query-level evaluation metrics and results.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from autorag_research.orm.repository import GenericRepository
from autorag_research.orm.schema import ExperimentResult


class ExperimentResultRepository(GenericRepository[ExperimentResult]):
    """Repository for ExperimentResult entity with composite key support."""

    def __init__(self, session: Session):
        """Initialize experiment result repository.

        Args:
            session: SQLAlchemy session for database operations.
        """
        super().__init__(session, ExperimentResult)

    def get_by_composite_key(self, query_id: int, pipeline_id: int, metric_id: int) -> ExperimentResult | None:
        """Retrieve an experiment result by its composite primary key.

        Args:
            query_id: The query ID.
            pipeline_id: The pipeline ID.
            metric_id: The metric ID.

        Returns:
            The experiment result if found, None otherwise.
        """
        stmt = select(ExperimentResult).where(
            ExperimentResult.query_id == query_id,
            ExperimentResult.pipeline_id == pipeline_id,
            ExperimentResult.metric_id == metric_id,
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def get_by_query_id(self, query_id: int) -> list[ExperimentResult]:
        """Retrieve all experiment results for a specific query.

        Args:
            query_id: The query ID.

        Returns:
            List of experiment results for the query.
        """
        stmt = select(ExperimentResult).where(ExperimentResult.query_id == query_id)
        return list(self.session.execute(stmt).scalars().all())

    def get_by_pipeline_id(self, pipeline_id: int) -> list[ExperimentResult]:
        """Retrieve all experiment results for a specific pipeline.

        Args:
            pipeline_id: The pipeline ID.

        Returns:
            List of experiment results for the pipeline.
        """
        stmt = select(ExperimentResult).where(ExperimentResult.pipeline_id == pipeline_id)
        return list(self.session.execute(stmt).scalars().all())

    def get_by_metric_id(self, metric_id: int) -> list[ExperimentResult]:
        """Retrieve all experiment results for a specific metric.

        Args:
            metric_id: The metric ID.

        Returns:
            List of experiment results for the metric.
        """
        stmt = select(ExperimentResult).where(ExperimentResult.metric_id == metric_id)
        return list(self.session.execute(stmt).scalars().all())

    def get_by_query_and_pipeline(self, query_id: int, pipeline_id: int) -> list[ExperimentResult]:
        """Retrieve all experiment results for a specific query and pipeline.

        Args:
            query_id: The query ID.
            pipeline_id: The pipeline ID.

        Returns:
            List of experiment results for the query and pipeline.
        """
        stmt = select(ExperimentResult).where(
            ExperimentResult.query_id == query_id,
            ExperimentResult.pipeline_id == pipeline_id,
        )
        return list(self.session.execute(stmt).scalars().all())

    def get_by_query_and_metric(self, query_id: int, metric_id: int) -> list[ExperimentResult]:
        """Retrieve all experiment results for a specific query and metric.

        Args:
            query_id: The query ID.
            metric_id: The metric ID.

        Returns:
            List of experiment results for the query and metric.
        """
        stmt = select(ExperimentResult).where(
            ExperimentResult.query_id == query_id,
            ExperimentResult.metric_id == metric_id,
        )
        return list(self.session.execute(stmt).scalars().all())

    def get_by_pipeline_and_metric(self, pipeline_id: int, metric_id: int) -> list[ExperimentResult]:
        """Retrieve all experiment results for a specific pipeline and metric.

        Args:
            pipeline_id: The pipeline ID.
            metric_id: The metric ID.

        Returns:
            List of experiment results for the pipeline and metric.
        """
        stmt = select(ExperimentResult).where(
            ExperimentResult.pipeline_id == pipeline_id,
            ExperimentResult.metric_id == metric_id,
        )
        return list(self.session.execute(stmt).scalars().all())

    def get_with_all_relations(self, query_id: int, pipeline_id: int, metric_id: int) -> ExperimentResult | None:
        """Retrieve an experiment result with all relations eagerly loaded.

        Args:
            query_id: The query ID.
            pipeline_id: The pipeline ID.
            metric_id: The metric ID.

        Returns:
            The experiment result with all relations loaded, None if not found.
        """
        stmt = (
            select(ExperimentResult)
            .where(
                ExperimentResult.query_id == query_id,
                ExperimentResult.pipeline_id == pipeline_id,
                ExperimentResult.metric_id == metric_id,
            )
            .options(
                joinedload(ExperimentResult.query_obj),
                joinedload(ExperimentResult.pipeline),
                joinedload(ExperimentResult.metric),
            )
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def get_with_generation_results(self, query_id: int, pipeline_id: int) -> list[ExperimentResult]:
        """Retrieve experiment results with non-null generation results.

        Args:
            query_id: The query ID.
            pipeline_id: The pipeline ID.

        Returns:
            List of experiment results with generation results.
        """
        stmt = select(ExperimentResult).where(
            ExperimentResult.query_id == query_id,
            ExperimentResult.pipeline_id == pipeline_id,
            ExperimentResult.generation_result.is_not(None),
        )
        return list(self.session.execute(stmt).scalars().all())

    def get_by_metric_result_range(
        self, pipeline_id: int, metric_id: int, min_result: float, max_result: float
    ) -> list[ExperimentResult]:
        """Retrieve experiment results within a metric result range.

        Args:
            pipeline_id: The pipeline ID.
            metric_id: The metric ID.
            min_result: Minimum metric result value (inclusive).
            max_result: Maximum metric result value (inclusive).

        Returns:
            List of experiment results within the specified range.
        """
        stmt = select(ExperimentResult).where(
            ExperimentResult.pipeline_id == pipeline_id,
            ExperimentResult.metric_id == metric_id,
            ExperimentResult.metric_result >= min_result,
            ExperimentResult.metric_result <= max_result,
        )
        return list(self.session.execute(stmt).scalars().all())

    def delete_by_composite_key(self, query_id: int, pipeline_id: int, metric_id: int) -> bool:
        """Delete an experiment result by its composite primary key.

        Args:
            query_id: The query ID.
            pipeline_id: The pipeline ID.
            metric_id: The metric ID.

        Returns:
            True if the result was deleted, False if not found.
        """
        result = self.get_by_composite_key(query_id, pipeline_id, metric_id)
        if result:
            self.delete(result)
            return True
        return False

    def exists_by_composite_key(self, query_id: int, pipeline_id: int, metric_id: int) -> bool:
        """Check if an experiment result exists with the given composite key.

        Args:
            query_id: The query ID.
            pipeline_id: The pipeline ID.
            metric_id: The metric ID.

        Returns:
            True if the result exists, False otherwise.
        """
        return self.get_by_composite_key(query_id, pipeline_id, metric_id) is not None
