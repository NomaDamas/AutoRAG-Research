"""Generation Evaluation Service for AutoRAG-Research.

Provides service layer for evaluating generation pipelines:
1. Fetch queries and ground truth from database
2. Fetch generation results (ExecutorResult)
3. Compute evaluation metrics
4. Store evaluation results
"""

import logging
from typing import Any

from autorag_research.orm.service.base_evaluation import BaseEvaluationService
from autorag_research.orm.uow.evaluation_uow import GenerationEvaluationUnitOfWork
from autorag_research.schema import MetricInput

__all__ = ["GenerationEvaluationService"]

logger = logging.getLogger("AutoRAG-Research")


class GenerationEvaluationService(BaseEvaluationService):
    """Service for evaluating generation pipelines.

    This service handles the evaluation workflow for generation pipelines:
    1. Fetch queries and ground truth (Query.generation_gt)
    2. Fetch generation results (ExecutorResult.generation_result)
    3. Compute evaluation metrics (e.g., BLEU, ROUGE, F1)
    4. Store results in EvaluationResult table

    The service uses MetricInput to pass data to metric functions, which should
    accept list[MetricInput] and return list[float | None].

    Example:
        ```python
        from autorag_research.orm.service import GenerationEvaluationService

        # Create service
        service = GenerationEvaluationService(session_factory, schema)

        # Get or create metric
        metric_id = service.get_or_create_metric("bleu", "generation")

        # Set metric and evaluate
        service.set_metric(metric_id=metric_id, metric_func=bleu_score)
        count, avg = service.evaluate(pipeline_id=1, batch_size=100)
        print(f"Evaluated {count} queries, average={avg}")
        ```
    """

    def _create_uow(self) -> GenerationEvaluationUnitOfWork:
        """Create a new GenerationEvaluationUnitOfWork instance.

        Returns:
            GenerationEvaluationUnitOfWork for managing evaluation transactions.
        """
        return GenerationEvaluationUnitOfWork(self.session_factory, self._schema)

    def _get_schema_classes(self) -> dict[str, Any]:
        """Get schema classes from the schema namespace.

        Returns:
            Dictionary mapping class names to ORM classes.
        """
        if self._schema is not None:
            return {
                "Query": self._schema.Query,
                "Pipeline": self._schema.Pipeline,
                "Metric": self._schema.Metric,
                "ExecutorResult": self._schema.ExecutorResult,
                "EvaluationResult": self._schema.EvaluationResult,
            }

        from autorag_research.orm.schema import (
            EvaluationResult,
            ExecutorResult,
            Metric,
            Pipeline,
            Query,
        )

        return {
            "Query": Query,
            "Pipeline": Pipeline,
            "Metric": Metric,
            "ExecutorResult": ExecutorResult,
            "EvaluationResult": EvaluationResult,
        }

    def _get_execution_results(self, pipeline_id: int, query_ids: list[int]) -> dict[int, dict[str, Any]]:
        """Fetch execution results for given query IDs.

        Fetches generation results (ExecutorResult) and ground truth
        (Query.generation_gt) for each query.

        Args:
            pipeline_id: The pipeline ID.
            query_ids: List of query IDs to fetch results for.

        Returns:
            Dictionary mapping query_id to dict with:
                - 'generated_text': generated text from ExecutorResult
                - 'generation_gt': list of ground truth texts from Query
        """
        with self._create_uow() as uow:
            result: dict[int, dict[str, Any]] = {}

            for query_id in query_ids:
                executor_result = uow.executor_results.get_by_composite_key(query_id, pipeline_id)
                if executor_result is None:
                    continue

                query = uow.queries.get_by_id(query_id)
                generation_gt = query.generation_gt if query else None

                result[query_id] = {
                    "generated_text": executor_result.generation_result,
                    "generation_gt": generation_gt,
                }

            return result

    def _filter_missing_query_ids(self, pipeline_id: int, metric_id: int, query_ids: list[int]) -> list[int]:
        """Filter query IDs that don't have evaluation results for the metric.

        Args:
            pipeline_id: The pipeline ID.
            metric_id: The metric ID.
            query_ids: List of query IDs to check.

        Returns:
            List of query IDs that need evaluation (don't have results yet).
        """
        with self._create_uow() as uow:
            existing_results = uow.evaluation_results.get_by_pipeline_and_metric(pipeline_id, metric_id)
            existing_query_ids = {r.query_id for r in existing_results}

            return [qid for qid in query_ids if qid not in existing_query_ids]

    def _prepare_metric_input(self, pipeline_id: int, query_id: int, execution_result: dict[str, Any]) -> MetricInput:
        """Prepare MetricInput for metric computation.

        Args:
            pipeline_id: The pipeline ID.
            query_id: The query ID.
            execution_result: Dict with 'generated_text' and 'generation_gt'.

        Returns:
            MetricInput instance ready for metric function.
        """
        return MetricInput(
            generated_texts=execution_result.get("generated_text"),
            generation_gt=execution_result.get("generation_gt"),
        )

    def _has_results_for_queries(self, pipeline_id: int, query_ids: list[int]) -> bool:
        """Check if all given query IDs have generation results for the pipeline.

        Checks ExecutorResult table for this pipeline.

        Args:
            pipeline_id: The pipeline ID.
            query_ids: List of query IDs to check.

        Returns:
            True if all query IDs have results, False otherwise.
        """
        with self._create_uow() as uow:
            for query_id in query_ids:
                result = uow.executor_results.get_by_composite_key(query_id, pipeline_id)
                if result is None:
                    return False

            return True
