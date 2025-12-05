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
    - IDs are prefixed with 'chunk_' or 'image_chunk_' to distinguish types

    Args:
        relations: List of RetrievalRelation entities with group_index, group_order,
                   chunk_id (optional), and image_chunk_id (optional).

    Returns:
        2D list of prefixed IDs: list[list[str]]
        - Outer list: AND conditions (all groups must be satisfied)
        - Inner list: OR conditions (any item in group satisfies)
        - IDs are prefixed: 'chunk_{id}' or 'image_chunk_{id}'

    Example:
        Relations with:
            (group_index=0, group_order=0, chunk_id=1)
            (group_index=0, group_order=1, image_chunk_id=2)
            (group_index=1, group_order=0, chunk_id=3)
        Returns: [["chunk_1", "image_chunk_2"], ["chunk_3"]]
        Meaning: (chunk_1 OR image_chunk_2) AND chunk_3
    """
    # Group by group_index, storing (group_order, prefixed_id) tuples
    grouped: dict[int, list[tuple[int, str]]] = defaultdict(list)

    for rel in relations:
        if rel.chunk_id is not None:
            prefixed_id = f"chunk_{rel.chunk_id}"
            grouped[rel.group_index].append((rel.group_order, prefixed_id))
        elif rel.image_chunk_id is not None:
            prefixed_id = f"image_chunk_{rel.image_chunk_id}"
            grouped[rel.group_index].append((rel.group_order, prefixed_id))

    # Sort each group by group_order, then extract prefixed_ids
    result: list[list[str]] = []
    for group_index in sorted(grouped.keys()):
        items = grouped[group_index]
        # Sort by group_order and extract prefixed_ids
        sorted_ids = [prefixed_id for _, prefixed_id in sorted(items, key=lambda x: x[0])]
        result.append(sorted_ids)

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
                "ImageChunkRetrievedResult": self._schema.ImageChunkRetrievedResult,
                "EvaluationResult": self._schema.EvaluationResult,
            }

        from autorag_research.orm.schema import (
            ChunkRetrievedResult,
            EvaluationResult,
            ImageChunkRetrievedResult,
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
            "ImageChunkRetrievedResult": ImageChunkRetrievedResult,
            "EvaluationResult": EvaluationResult,
        }

    def _get_execution_results(self, pipeline_id: int, query_ids: list[int]) -> dict[int, dict[str, Any]]:
        """Fetch execution results for given query IDs.

        Fetches both retrieval results (ChunkRetrievedResult and ImageChunkRetrievedResult)
        and ground truth (RetrievalRelation) for each query.

        Args:
            pipeline_id: The pipeline ID.
            query_ids: List of query IDs to fetch results for.

        Returns:
            Dictionary mapping query_id to dict with:
                - 'retrieved_ids': list of prefixed retrieved IDs (ordered by rel_score desc)
                  Format: 'chunk_{id}' or 'image_chunk_{id}'
                - 'retrieval_gt': 2D list of prefixed ground truth IDs (grouped by AND/OR conditions)
        """
        with self._create_uow() as uow:
            result: dict[int, dict[str, Any]] = {}

            for query_id in query_ids:
                # Get chunk retrieval results ordered by rel_score descending
                chunk_results = uow.chunk_results.get_by_query_and_pipeline(query_id, pipeline_id)
                chunk_ids = [(r.rel_score or 0.0, f"chunk_{r.chunk_id}") for r in chunk_results]

                # Get image chunk retrieval results ordered by rel_score descending
                image_chunk_results = uow.image_chunk_results.get_by_query_and_pipeline(query_id, pipeline_id)
                image_chunk_ids = [(r.rel_score or 0.0, f"image_chunk_{r.image_chunk_id}") for r in image_chunk_results]

                # Merge and sort by rel_score descending
                all_results = chunk_ids + image_chunk_ids
                all_results.sort(key=lambda x: x[0], reverse=True)
                retrieved_ids = [prefixed_id for _, prefixed_id in all_results]

                # Get ground truth as 2D list (AND/OR structure) with prefixes
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
            # Get existing evaluation results for this pipeline and metric
            existing_results = uow.evaluation_results.get_by_pipeline_and_metric(pipeline_id, metric_id)
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
