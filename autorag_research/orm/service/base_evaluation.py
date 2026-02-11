"""Base Evaluation Service for AutoRAG-Research.

Provides abstract base class for evaluation services with:
- Metric management (get, create, set)
- Batch metric computation
- Pipeline for fetching, computing, and saving evaluation results
- Generator-based pagination to avoid loading all query IDs at once
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from typing import Any, TypeVar

from sqlalchemy.orm import Session, sessionmaker

from autorag_research.exceptions import NoQueryInDBError, SchemaNotFoundError
from autorag_research.orm.service.base import BaseService
from autorag_research.schema import MetricInput

logger = logging.getLogger("AutoRAG-Research")

T = TypeVar("T")
R = TypeVar("R")

# Type alias for metric function
# Accepts list[MetricInput] and returns list[float | None]
MetricFunc = Callable[[list[MetricInput]], list[float | None]]


class BaseEvaluationService(BaseService, ABC):
    """Abstract base class for evaluation services.

    Provides common patterns for evaluation workflows:
    1. Fetch execution results in batches using Generator (abstract)
    2. Filter missing query IDs that need evaluation (abstract)
    3. Compute metrics with batch processing (base)
    4. Save evaluation results (abstract)

    The service supports:
    - Setting and changing metric functions dynamically
    - Batch processing with configurable batch size
    - Generator-based pagination to minimize memory usage and transaction issues

    Example:
        ```python
        service = RetrievalEvaluationService(session_factory, schema)

        # Set metric and evaluate
        service.set_metric(metric_id=1, metric_func=my_metric_func)
        service.evaluate(pipeline_id=1, batch_size=100)

        # Change metric and evaluate again
        service.set_metric(metric_id=2, metric_func=another_metric_func)
        service.evaluate(pipeline_id=1, batch_size=100)
        ```
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        schema: Any | None = None,
    ):
        """Initialize the evaluation service.

        Args:
            session_factory: SQLAlchemy sessionmaker for database connections.
            schema: Schema namespace from create_schema(). If None, uses default 768-dim schema.
        """
        super().__init__(session_factory, schema)
        self._metric_id: int | str | None = None
        self._metric_func: MetricFunc | None = None

    @property
    def metric_id(self) -> int | str | None:
        """Get current metric ID."""
        return self._metric_id

    @property
    def metric_func(self) -> MetricFunc | None:
        """Get current metric function."""
        return self._metric_func

    def set_metric(self, metric_id: int | str, metric_func: MetricFunc) -> None:
        """Set the metric ID and function for evaluation.

        Args:
            metric_id: The ID of the metric in the database.
            metric_func: Function that takes list[MetricInput] and returns list[float | None].
        """
        self._metric_id = metric_id
        self._metric_func = metric_func

    def get_metric(self, metric_name: str, metric_type: str | None = None) -> Any | None:
        """Get metric by name and optionally type.

        Args:
            metric_name: The name of the metric.
            metric_type: Optional metric type filter ('retrieval' or 'generation').

        Returns:
            The Metric entity if found, None otherwise.
        """
        with self._create_uow() as uow:
            if metric_type:
                return uow.metrics.get_by_name_and_type(metric_name, metric_type)
            return uow.metrics.get_by_name(metric_name)

    def get_or_create_metric(self, name: str, metric_type: str) -> int | str:
        """Get existing metric or create a new one.

        Args:
            name: The metric name.
            metric_type: The metric type ('retrieval' or 'generation').

        Returns:
            The metric ID.
        """
        classes = self._get_schema_classes()
        metric_cls = classes.get("Metric")
        if metric_cls is None:
            raise SchemaNotFoundError("Metric")

        with self._create_uow() as uow:
            existing = uow.metrics.get_by_name_and_type(name, metric_type)
            if existing:
                return existing.id

            metric = metric_cls(name=name, type=metric_type)
            uow.metrics.add(metric)
            uow.flush()
            metric_id = metric.id
            uow.commit()
            return metric_id

    def _iter_query_id_batches(self, batch_size: int) -> Generator[list[int | str], None, None]:
        """Iterate over query IDs in batches using pagination.

        This method uses limit/offset to fetch query IDs in batches,
        avoiding loading all IDs into memory at once.

        Args:
            batch_size: Number of query IDs per batch.

        Yields:
            List of query IDs for each batch.
        """
        offset = 0
        while True:
            batch = self._fetch_query_ids_batch(batch_size, offset)
            if not batch:
                break
            yield batch
            offset += batch_size

    def _fetch_query_ids_batch(self, limit: int, offset: int) -> list[int | str]:
        """Fetch a batch of query IDs with pagination.

        Uses QueryRepository to get all query IDs ordered by ID.

        Args:
            limit: Maximum number of query IDs to fetch.
            offset: Number of query IDs to skip.

        Returns:
            List of query IDs for this batch.
        """
        with self._create_uow() as uow:
            return uow.queries.get_all_ids(limit, offset)

    def _count_total_query_ids(self) -> int:
        """Count total number of queries.

        Returns:
            Total count of queries.
        """
        with self._create_uow() as uow:
            return uow.queries.count_all()

    @abstractmethod
    def _get_execution_results(self, pipeline_id: int | str, query_ids: list[int | str]) -> dict[int | str, Any]:
        """Fetch execution results for given query IDs.

        Args:
            pipeline_id: The pipeline ID.
            query_ids: List of query IDs to fetch results for.

        Returns:
            Dictionary mapping query_id to execution result data.
        """
        ...

    @abstractmethod
    def _filter_missing_query_ids(
        self, pipeline_id: int | str, metric_id: int | str, query_ids: list[int | str]
    ) -> list[int | str]:
        """Filter query IDs that don't have evaluation results for the metric.

        Args:
            pipeline_id: The pipeline ID.
            metric_id: The metric ID.
            query_ids: List of query IDs to check.

        Returns:
            List of query IDs that need evaluation.
        """
        ...

    @abstractmethod
    def _prepare_metric_input(self, pipeline_id: int | str, query_id: int | str, execution_result: Any) -> MetricInput:
        """Prepare input data for metric computation.

        Args:
            pipeline_id: The pipeline ID.
            query_id: The query ID.
            execution_result: The execution result data.

        Returns:
            MetricInput instance ready for metric function.
        """
        ...

    def _save_evaluation_results(
        self, pipeline_id: int | str, metric_id: int | str, results: list[tuple[int | str, float]]
    ) -> None:
        """Save computed evaluation results to the database.

        Args:
            pipeline_id: The pipeline ID.
            metric_id: The metric ID.
            results: List of (query_id, metric_score) tuples.
        """
        classes = self._get_schema_classes()
        eval_result_cls = classes.get("EvaluationResult")
        if eval_result_cls is None:
            raise SchemaNotFoundError("EvaluationResult")

        with self._create_uow() as uow:
            entities = [
                eval_result_cls(
                    query_id=query_id,
                    pipeline_id=pipeline_id,
                    metric_id=metric_id,
                    metric_result=score,
                )
                for query_id, score in results
            ]
            uow.evaluation_results.add_all(entities)
            uow.commit()

            logger.debug(
                f"Saved {len(results)} evaluation results for pipeline_id={pipeline_id}, metric_id={metric_id}"
            )

    def _compute_metrics_batch(
        self,
        query_ids: list[int | str],
        metric_inputs: list[MetricInput],
    ) -> list[tuple[int | str, float | None]]:
        """Compute metrics for a batch of items.

        Args:
            query_ids: List of query IDs corresponding to metric inputs.
            metric_inputs: List of MetricInput instances.

        Returns:
            List of (query_id, metric_score) tuples. Score is None if computation failed.
        """
        if self._metric_func is None:
            raise ValueError("Metric function not set. Call set_metric() first.")  # noqa: TRY003

        try:
            scores = self._metric_func(metric_inputs)
            return list(zip(query_ids, scores, strict=True))
        except Exception:
            logger.exception("Failed to compute metrics for batch")
            return [(query_id, None) for query_id in query_ids]

    def evaluate(
        self,
        pipeline_id: int | str,
        batch_size: int = 100,
    ) -> tuple[int, float | None]:
        """Run the full evaluation pipeline for the current metric.

        This method uses Generator-based pagination to process query IDs:
        1. Iterates through query ID batches using limit/offset
        2. Filters to only those missing evaluation results
        3. Fetches execution results for the batch
        4. Computes metrics in batch
        5. Saves results to database

        Args:
            pipeline_id: The pipeline ID to evaluate.
            batch_size: Number of queries to process per batch.

        Returns:
            Tuple of (queries_evaluated, average_score).
            average_score is None if no queries were evaluated.

        Raises:
            ValueError: If metric is not set.
        """
        if self._metric_id is None or self._metric_func is None:
            raise ValueError("Metric not set. Call set_metric() first.")  # noqa: TRY003

        total_count = self._count_total_query_ids()
        if total_count == 0:
            logger.exception("No queries found for evaluation")
            raise NoQueryInDBError

        logger.info(
            f"Starting evaluation for pipeline_id={pipeline_id}, "
            f"metric_id={self._metric_id}, total_queries={total_count}"
        )

        total_evaluated = 0
        all_scores: list[float] = []

        for batch_num, batch_query_ids in enumerate(self._iter_query_id_batches(batch_size)):
            # Filter to missing query IDs for this batch
            missing_query_ids = self._filter_missing_query_ids(pipeline_id, self._metric_id, batch_query_ids)

            if not missing_query_ids:
                logger.debug(f"Batch {batch_num}: All queries already evaluated, skipping")
                continue

            # Fetch execution results for missing queries
            execution_results = self._get_execution_results(pipeline_id, missing_query_ids)

            # Prepare metric inputs
            query_ids: list[int | str] = []
            metric_inputs: list[MetricInput] = []
            for query_id in missing_query_ids:
                if query_id in execution_results:
                    metric_input = self._prepare_metric_input(pipeline_id, query_id, execution_results[query_id])
                    query_ids.append(query_id)
                    metric_inputs.append(metric_input)

            if not metric_inputs:
                continue

            # Compute metrics in batch
            results = self._compute_metrics_batch(query_ids, metric_inputs)

            # Filter successful results
            valid_results = [(query_id, score) for query_id, score in results if score is not None]

            # Save results and collect scores for average
            if valid_results:
                self._save_evaluation_results(pipeline_id, self._metric_id, valid_results)
                total_evaluated += len(valid_results)
                all_scores.extend(score for _, score in valid_results)

            logger.info(f"Batch {batch_num}: Evaluated {len(valid_results)}/{len(metric_inputs)} queries")

        # Calculate average
        average = sum(all_scores) / len(all_scores) if all_scores else None

        logger.info(
            f"Evaluation complete: {total_evaluated} queries evaluated for "
            f"pipeline_id={pipeline_id}, metric_id={self._metric_id}, average={average}"
        )
        return total_evaluated, average

    def is_evaluation_complete(
        self,
        pipeline_id: int | str,
        metric_id: int | str,
        batch_size: int = 100,
    ) -> bool:
        """Check if evaluation is complete for all queries.

        Iterates through all query IDs and checks:
        1. Each query has execution results
        2. Each query has evaluation results for the given pipeline and metric

        Args:
            pipeline_id: The pipeline ID to check.
            metric_id: The metric ID to check.
            batch_size: Number of queries to check per batch.

        Returns:
            True if all queries have both execution and evaluation results,
            False otherwise (returns immediately on first missing).
        """
        for batch_query_ids in self._iter_query_id_batches(batch_size):
            # Check if all queries have evaluation results
            missing = self._filter_missing_query_ids(pipeline_id, metric_id, batch_query_ids)
            if missing:
                return False

        return True

    def verify_pipeline_completion(self, pipeline_id: int | str, batch_size: int = 100) -> bool:
        """Verify all queries have execution results for the pipeline.

        Iterates through query IDs in batches and checks each batch has results.
        Returns False immediately when any query is missing results.

        Args:
            pipeline_id: The pipeline ID to verify.
            batch_size: Number of queries to check per batch.

        Returns:
            True if all queries have results, False otherwise.

        Raises:
            NoQueryInDBError: If no queries exist in the database.
        """
        total_queries = self._count_total_query_ids()

        if total_queries == 0:
            raise NoQueryInDBError

        for batch_query_ids in self._iter_query_id_batches(batch_size):
            if not self._has_results_for_queries(pipeline_id, batch_query_ids):
                return False

        logger.debug(f"Pipeline {pipeline_id} verified: all {total_queries} queries have results")
        return True

    @abstractmethod
    def _has_results_for_queries(self, pipeline_id: int | str, query_ids: list[int | str]) -> bool:
        """Check if all given query IDs have execution results for the pipeline.

        Args:
            pipeline_id: The pipeline ID.
            query_ids: List of query IDs to check.

        Returns:
            True if all query IDs have results, False otherwise.
        """
        ...
