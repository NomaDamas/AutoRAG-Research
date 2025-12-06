"""Retrieval Evaluation Service for AutoRAG-Research.

Provides service layer for evaluating retrieval pipelines:
1. Fetch queries and ground truth from database
2. Fetch retrieval results
3. Compute evaluation metrics
4. Store evaluation results
"""

import logging
from collections import defaultdict
from typing import Any

from autorag_research.exceptions import SchemaNotFoundError
from autorag_research.orm.service.base_evaluation import BaseEvaluationService
from autorag_research.orm.uow.evaluation_uow import RetrievalEvaluationUnitOfWork
from autorag_research.schema import MetricInput

__all__ = ["RetrievalEvaluationService", "build_retrieval_gt_from_relations"]

logger = logging.getLogger("AutoRAG-Research")


def build_retrieval_gt_from_relations(relations: list[Any]) -> list[list[str]]:
    """Build 2D retrieval ground truth list from RetrievalRelation entities.

    Converts RetrievalRelation entities into a 2D list structure for metric computation.
    - Same group_index = OR condition (items in same inner list)
    - Different group_index = AND condition (items in different inner lists)
    - Items within each group are ordered by group_order

    Args:
        relations: List of RetrievalRelation entities with group_index, group_order, chunk_id.

    Returns:
        2D list of chunk IDs: list[list[str]]
        - Outer list: AND conditions (all groups must be satisfied)
        - Inner list: OR conditions (any item in group satisfies)

    Example:
        Relations with:
            (group_index=0, group_order=0, chunk_id=1)
            (group_index=0, group_order=1, chunk_id=2)
            (group_index=1, group_order=0, chunk_id=3)
        Returns: [["1", "2"], ["3"]]
        Meaning: (chunk_1 OR chunk_2) AND chunk_3
    """
    # Group by group_index, storing (group_order, chunk_id) tuples
    grouped: dict[int, list[tuple[int, str]]] = defaultdict(list)

    for rel in relations:
        if rel.chunk_id is not None:
            grouped[rel.group_index].append((rel.group_order, str(rel.chunk_id)))

    # Sort each group by group_order, then extract chunk_ids
    result: list[list[str]] = []
    for group_index in sorted(grouped.keys()):
        items = grouped[group_index]
        # Sort by group_order and extract chunk_ids
        sorted_chunk_ids = [chunk_id for _, chunk_id in sorted(items, key=lambda x: x[0])]
        result.append(sorted_chunk_ids)

    return result


class RetrievalEvaluationService(BaseEvaluationService):
    """Service for evaluating retrieval pipelines.

    This service handles the evaluation workflow for retrieval pipelines:
    1. Fetch queries and ground truth (RetrievalRelation)
    2. Fetch retrieval results (ChunkRetrievedResult)
    3. Compute evaluation metrics (e.g., Recall@K, Precision@K, MRR)
    4. Store results in EvaluationResult table

    The service uses MetricInput to pass data to metric functions, which should
    accept MetricInput and return a float score.

    Example:
        ```python
        from autorag_research.orm.service import RetrievalEvaluationService
        from autorag_research.evaluation.metrics.retrieval import retrieval_recall

        # Create service
        service = RetrievalEvaluationService(session_factory, schema)

        # Get or create metric
        metric_id = service.get_or_create_metric("recall@10", "retrieval")

        # Set metric and evaluate
        service.set_metric(metric_id=metric_id, metric_func=retrieval_recall)
        count = service.evaluate(
            pipeline_id=1,
            batch_size=100,
            max_concurrent=10,
        )
        print(f"Evaluated {count} queries")

        # Evaluate another metric
        metric_id_2 = service.get_or_create_metric("precision@10", "retrieval")
        service.set_metric(metric_id=metric_id_2, metric_func=retrieval_precision)
        service.evaluate(pipeline_id=1)
        ```
    """

    def _create_uow(self) -> RetrievalEvaluationUnitOfWork:
        """Create a new RetrievalEvaluationUnitOfWork instance.

        Returns:
            RetrievalEvaluationUnitOfWork for managing evaluation transactions.
        """
        return RetrievalEvaluationUnitOfWork(self.session_factory, self._schema)

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
                "RetrievalRelation": self._schema.RetrievalRelation,
                "ChunkRetrievedResult": self._schema.ChunkRetrievedResult,
                "EvaluationResult": self._schema.EvaluationResult,
            }

        from autorag_research.orm.schema import (
            ChunkRetrievedResult,
            EvaluationResult,
            Metric,
            Pipeline,
            Query,
            RetrievalRelation,
        )

        return {
            "Query": Query,
            "Pipeline": Pipeline,
            "Metric": Metric,
            "RetrievalRelation": RetrievalRelation,
            "ChunkRetrievedResult": ChunkRetrievedResult,
            "EvaluationResult": EvaluationResult,
        }

    def _get_execution_results(self, pipeline_id: int, query_ids: list[int]) -> dict[int, dict[str, Any]]:
        """Fetch execution results for given query IDs.

        Fetches both retrieval results (ChunkRetrievedResult) and ground truth
        (RetrievalRelation) for each query.

        Args:
            pipeline_id: The pipeline ID.
            query_ids: List of query IDs to fetch results for.

        Returns:
            Dictionary mapping query_id to dict with:
                - 'retrieved_ids': list of retrieved chunk IDs (ordered by rel_score desc)
                - 'retrieval_gt': 2D list of ground truth chunk IDs (grouped by AND/OR conditions)
        """
        with self._create_uow() as uow:
            result: dict[int, dict[str, Any]] = {}

            # Fetch all chunk results at once (already sorted by query_id asc, rel_score desc)
            chunk_results = uow.chunk_results.get_by_query_and_pipeline(query_ids, pipeline_id)

            # Group chunk results by query_id
            chunk_results_by_query: dict[int, list[Any]] = {qid: [] for qid in query_ids}
            for r in chunk_results:
                if r.query_id in chunk_results_by_query:
                    chunk_results_by_query[r.query_id].append(r)

            for query_id in query_ids:
                # Get retrieval results (already ordered by rel_score descending)
                retrieved_ids = [str(r.chunk_id) for r in chunk_results_by_query[query_id]]

                # Get ground truth as 2D list (AND/OR structure)
                gt_relations = uow.retrieval_relations.get_by_query_id(query_id)
                retrieval_gt = build_retrieval_gt_from_relations(gt_relations)

                result[query_id] = {
                    "retrieved_ids": retrieved_ids,
                    "retrieval_gt": retrieval_gt,
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
            # Get existing evaluation results filtered by pipeline, metric, AND query_ids
            existing_results = uow.evaluation_results.get_by_pipeline_metric_and_queries(
                pipeline_id, metric_id, query_ids
            )
            existing_query_ids = {r.query_id for r in existing_results}

            # Return query IDs that don't have results
            return [qid for qid in query_ids if qid not in existing_query_ids]

    def _prepare_metric_input(self, pipeline_id: int, query_id: int, execution_result: dict[str, Any]) -> MetricInput:
        """Prepare MetricInput for metric computation.

        Args:
            pipeline_id: The pipeline ID.
            query_id: The query ID.
            execution_result: Dict with 'retrieved_ids' and 'retrieval_gt'.

        Returns:
            MetricInput instance ready for metric function.
        """
        return MetricInput(
            retrieved_ids=execution_result.get("retrieved_ids"),
            retrieval_gt=execution_result.get("retrieval_gt"),
        )

    def _save_evaluation_results(self, pipeline_id: int, metric_id: int, results: list[tuple[int, float]]) -> None:
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
