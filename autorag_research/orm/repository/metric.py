"""Metric repository for AutoRAG-Research.

Implements metric-specific CRUD operations and relationship queries
for managing evaluation metrics and their results.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from autorag_research.orm.repository.base import GenericRepository
from autorag_research.orm.schema import Metric


class MetricRepository(GenericRepository[Metric]):
    """Repository for Metric entity with relationship loading capabilities."""

    def __init__(self, session: Session):
        """Initialize metric repository.

        Args:
            session: SQLAlchemy session for database operations.
        """
        super().__init__(session, Metric)

    def get_by_name(self, name: str) -> Metric | None:
        """Retrieve a metric by its name.

        Args:
            name: The metric name to search for.

        Returns:
            The metric if found, None otherwise.
        """
        stmt = select(Metric).where(Metric.name == name)
        return self.session.execute(stmt).scalar_one_or_none()

    def get_by_name_and_type(self, name: str, metric_type: str) -> Metric | None:
        """Retrieve a metric by its name and type.

        Args:
            name: The metric name to search for.
            metric_type: The metric type (retrieval or generation).

        Returns:
            The metric if found, None otherwise.
        """
        stmt = select(Metric).where(Metric.name == name, Metric.type == metric_type)
        return self.session.execute(stmt).scalar_one_or_none()

    def get_by_type(self, metric_type: str) -> list[Metric]:
        """Retrieve all metrics of a specific type.

        Args:
            metric_type: The metric type (retrieval or generation).

        Returns:
            List of metrics of the specified type.
        """
        stmt = select(Metric).where(Metric.type == metric_type).order_by(Metric.name)
        return list(self.session.execute(stmt).scalars().all())

    def get_with_experiment_results(self, metric_id: int) -> Metric | None:
        """Retrieve a metric with its experiment results eagerly loaded.

        Args:
            metric_id: The metric ID.

        Returns:
            The metric with experiment results loaded, None if not found.
        """
        stmt = select(Metric).where(Metric.id == metric_id).options(joinedload(Metric.experiment_results))
        return self.session.execute(stmt).scalar_one_or_none()

    def get_with_summaries(self, metric_id: int) -> Metric | None:
        """Retrieve a metric with its summaries eagerly loaded.

        Args:
            metric_id: The metric ID.

        Returns:
            The metric with summaries loaded, None if not found.
        """
        stmt = select(Metric).where(Metric.id == metric_id).options(joinedload(Metric.summaries))
        return self.session.execute(stmt).scalar_one_or_none()

    def get_with_retrieved_results(self, metric_id: int) -> Metric | None:
        """Retrieve a metric with its retrieved results eagerly loaded.

        Args:
            metric_id: The metric ID.

        Returns:
            The metric with chunk and image chunk retrieved results loaded, None if not found.
        """
        stmt = (
            select(Metric)
            .where(Metric.id == metric_id)
            .options(
                joinedload(Metric.chunk_retrieved_results),
                joinedload(Metric.image_chunk_retrieved_results),
            )
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def get_with_all_relations(self, metric_id: int) -> Metric | None:
        """Retrieve a metric with all relations eagerly loaded.

        Args:
            metric_id: The metric ID.

        Returns:
            The metric with all relations loaded, None if not found.
        """
        stmt = (
            select(Metric)
            .where(Metric.id == metric_id)
            .options(
                joinedload(Metric.experiment_results),
                joinedload(Metric.summaries),
                joinedload(Metric.chunk_retrieved_results),
                joinedload(Metric.image_chunk_retrieved_results),
            )
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def search_by_name(self, search_text: str, limit: int = 10) -> list[Metric]:
        """Search metrics containing the specified text in their name.

        Args:
            search_text: Text to search for in metric names.
            limit: Maximum number of results to return.

        Returns:
            List of metrics containing the search text.
        """
        stmt = select(Metric).where(Metric.name.ilike(f"%{search_text}%")).limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def get_all_retrieval_metrics(self) -> list[Metric]:
        """Retrieve all retrieval metrics.

        Returns:
            List of all retrieval metrics ordered by name.
        """
        return self.get_by_type("retrieval")

    def get_all_generation_metrics(self) -> list[Metric]:
        """Retrieve all generation metrics.

        Returns:
            List of all generation metrics ordered by name.
        """
        return self.get_by_type("generation")

    def exists_by_name(self, name: str) -> bool:
        """Check if a metric exists with the given name.

        Args:
            name: The metric name to check.

        Returns:
            True if a metric exists, False otherwise.
        """
        return self.get_by_name(name) is not None

    def exists_by_name_and_type(self, name: str, metric_type: str) -> bool:
        """Check if a metric exists with the given name and type.

        Args:
            name: The metric name to check.
            metric_type: The metric type (retrieval or generation).

        Returns:
            True if a metric exists, False otherwise.
        """
        return self.get_by_name_and_type(name, metric_type) is not None
