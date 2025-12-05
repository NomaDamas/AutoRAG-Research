"""EvaluatorResult repository for AutoRAG-Research.

Implements evaluator result-specific CRUD operations and relationship queries
for managing query-pipeline-metric evaluation results.
"""

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from autorag_research.orm.repository.base import GenericRepository


class EvaluatorResultRepository(GenericRepository[Any]):
    """Repository for EvaluationResult entity with composite key support."""

    def __init__(self, session: Session, model_cls: type | None = None):
        """Initialize evaluator result repository.

        Args:
            session: SQLAlchemy session for database operations.
            model_cls: The EvaluationResult model class to use. If None, uses default schema.
        """
        if model_cls is None:
            from autorag_research.orm.schema import EvaluationResult

            model_cls = EvaluationResult
        super().__init__(session, model_cls)

    def get_by_composite_key(self, query_id: int, pipeline_id: int, metric_id: int) -> Any | None:
        """Retrieve an evaluation result by its composite primary key.

        Args:
            query_id: The query ID.
            pipeline_id: The pipeline ID.
            metric_id: The metric ID.

        Returns:
            The evaluation result if found, None otherwise.
        """
        stmt = select(self.model_cls).where(
            self.model_cls.query_id == query_id,
            self.model_cls.pipeline_id == pipeline_id,
            self.model_cls.metric_id == metric_id,
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def get_by_query_id(self, query_id: int) -> list[Any]:
        """Retrieve all evaluation results for a specific query.

        Args:
            query_id: The query ID.

        Returns:
            List of evaluation results for the query.
        """
        stmt = select(self.model_cls).where(self.model_cls.query_id == query_id)
        return list(self.session.execute(stmt).scalars().all())

    def get_by_pipeline_id(self, pipeline_id: int) -> list[Any]:
        """Retrieve all evaluation results for a specific pipeline.

        Args:
            pipeline_id: The pipeline ID.

        Returns:
            List of evaluation results for the pipeline.
        """
        stmt = select(self.model_cls).where(self.model_cls.pipeline_id == pipeline_id)
        return list(self.session.execute(stmt).scalars().all())

    def get_by_metric_id(self, metric_id: int) -> list[Any]:
        """Retrieve all evaluation results for a specific metric.

        Args:
            metric_id: The metric ID.

        Returns:
            List of evaluation results for the metric.
        """
        stmt = select(self.model_cls).where(self.model_cls.metric_id == metric_id)
        return list(self.session.execute(stmt).scalars().all())

    def get_by_query_and_pipeline(self, query_id: int, pipeline_id: int) -> list[Any]:
        """Retrieve all evaluation results for a specific query and pipeline.

        Args:
            query_id: The query ID.
            pipeline_id: The pipeline ID.

        Returns:
            List of evaluation results for the query-pipeline combination.
        """
        stmt = select(self.model_cls).where(
            self.model_cls.query_id == query_id,
            self.model_cls.pipeline_id == pipeline_id,
        )
        return list(self.session.execute(stmt).scalars().all())

    def get_with_all_relations(self, query_id: int, pipeline_id: int, metric_id: int) -> Any | None:
        """Retrieve an evaluation result with all relations eagerly loaded.

        Args:
            query_id: The query ID.
            pipeline_id: The pipeline ID.
            metric_id: The metric ID.

        Returns:
            The evaluation result with all relations loaded, None if not found.
        """
        stmt = (
            select(self.model_cls)
            .where(
                self.model_cls.query_id == query_id,
                self.model_cls.pipeline_id == pipeline_id,
                self.model_cls.metric_id == metric_id,
            )
            .options(
                joinedload(self.model_cls.query_obj),
                joinedload(self.model_cls.pipeline),
                joinedload(self.model_cls.metric),
            )
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def get_with_non_null_metric_result(self, query_id: int, pipeline_id: int, metric_id: int) -> Any | None:
        """Retrieve evaluation result with non-null metric result.

        Args:
            query_id: The query ID.
            pipeline_id: The pipeline ID.
            metric_id: The metric ID.

        Returns:
            The evaluation result if it has a metric result, None otherwise.
        """
        stmt = select(self.model_cls).where(
            self.model_cls.query_id == query_id,
            self.model_cls.pipeline_id == pipeline_id,
            self.model_cls.metric_id == metric_id,
            self.model_cls.metric_result.is_not(None),
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def get_by_metric_result_range(self, metric_id: int, min_score: float, max_score: float) -> list[Any]:
        """Retrieve evaluation results within a metric result range.

        Args:
            metric_id: The metric ID.
            min_score: Minimum metric result (inclusive).
            max_score: Maximum metric result (inclusive).

        Returns:
            List of evaluation results within the specified range.
        """
        stmt = select(self.model_cls).where(
            self.model_cls.metric_id == metric_id,
            self.model_cls.metric_result >= min_score,
            self.model_cls.metric_result <= max_score,
        )
        return list(self.session.execute(stmt).scalars().all())

    def get_by_pipeline_and_metric(self, pipeline_id: int, metric_id: int) -> list[Any]:
        """Retrieve all evaluation results for a specific pipeline and metric.

        Args:
            pipeline_id: The pipeline ID.
            metric_id: The metric ID.

        Returns:
            List of evaluation results for the pipeline-metric combination.
        """
        stmt = select(self.model_cls).where(
            self.model_cls.pipeline_id == pipeline_id,
            self.model_cls.metric_id == metric_id,
        )
        return list(self.session.execute(stmt).scalars().all())

    def delete_by_composite_key(self, query_id: int, pipeline_id: int, metric_id: int) -> bool:
        """Delete an evaluation result by its composite primary key.

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
        """Check if an evaluation result exists with the given composite key.

        Args:
            query_id: The query ID.
            pipeline_id: The pipeline ID.
            metric_id: The metric ID.

        Returns:
            True if the result exists, False otherwise.
        """
        return self.get_by_composite_key(query_id, pipeline_id, metric_id) is not None
