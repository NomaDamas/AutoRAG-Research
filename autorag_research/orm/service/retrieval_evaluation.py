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


def build_retrieval_gt_from_relations(relations: list[Any]) -> tuple[list[list[str]], dict[str, int]]:
    """Build 2D retrieval ground truth list and relevance scores from RetrievalRelation entities.

    Converts RetrievalRelation entities into a 2D list structure for metric computation.
    - Same group_index = OR condition (items in same inner list)
    - Different group_index = AND condition (items in different inner lists)
    - Items within each group are ordered by group_order
    - IDs are prefixed with 'chunk_' or 'image_chunk_' to distinguish types

    Args:
        relations: List of RetrievalRelation entities with group_index, group_order,
                   chunk_id (optional), image_chunk_id (optional), and score (optional).

    Returns:
        Tuple of (retrieval_gt, relevance_scores):
        - retrieval_gt: 2D list of prefixed IDs
          - Outer list: AND conditions (all groups must be satisfied)
          - Inner list: OR conditions (any item in group satisfies)
          - IDs are prefixed: 'chunk_{id}' or 'image_chunk_{id}'
        - relevance_scores: dict mapping prefixed_id -> score (default 1 if None)

    Example:
        Relations with:
            (group_index=0, group_order=0, chunk_id=1, score=2)
            (group_index=0, group_order=1, image_chunk_id=2, score=1)
            (group_index=1, group_order=0, chunk_id=3, score=None)
        Returns:
            ([["chunk_1", "image_chunk_2"], ["chunk_3"]], {"chunk_1": 2, "image_chunk_2": 1, "chunk_3": 1})
        Meaning: (chunk_1 OR image_chunk_2) AND chunk_3
    """
    # Group by group_index, storing (group_order, prefixed_id, score) tuples
    grouped: dict[int, list[tuple[int, str, int]]] = defaultdict(list)
    relevance_scores: dict[str, int] = {}

    for rel in relations:
        prefixed_id: str | None = None
        if rel.chunk_id is not None:
            prefixed_id = f"chunk_{rel.chunk_id}"
        elif rel.image_chunk_id is not None:
            prefixed_id = f"image_chunk_{rel.image_chunk_id}"

        if prefixed_id is not None:
            # Default score to 1 if None (backward compatibility)
            score = rel.score if rel.score is not None else 1
            grouped[rel.group_index].append((rel.group_order, prefixed_id, score))
            relevance_scores[prefixed_id] = score

    # Sort each group by group_order, then extract prefixed_ids
    result: list[list[str]] = []
    for group_index in sorted(grouped.keys()):
        items = grouped[group_index]
        # Sort by group_order and extract prefixed_ids
        sorted_ids = [prefixed_id for _, prefixed_id, _ in sorted(items, key=lambda x: x[0])]
        result.append(sorted_ids)

    return result, relevance_scores


class RetrievalEvaluationService(BaseEvaluationService):
    """Service for evaluating retrieval pipelines.

    This service handles the evaluation workflow for retrieval pipelines:
    1. Fetch queries and ground truth (RetrievalRelation)
    2. Fetch retrieval results (ChunkRetrievedResult)
    3. Compute evaluation metrics (e.g., Recall@K, Precision@K, MRR)
    4. Store results in EvaluationResult table

    The service uses MetricInput to pass data to metric functions, which should
    accept list[MetricInput] and return list[float | None].

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
        count, avg = service.evaluate(pipeline_id=1, batch_size=100)
        print(f"Evaluated {count} queries, average={avg}")

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

    def _get_execution_results(
        self, pipeline_id: int | str, query_ids: list[int | str]
    ) -> dict[int | str, dict[str, Any]]:
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
                - 'relevance_scores': dict mapping prefixed_id -> graded relevance score
        """
        with self._create_uow() as uow:
            result: dict[int | str, dict[str, Any]] = {}

            chunk_results = uow.chunk_results.get_by_query_and_pipeline(query_ids, pipeline_id)
            image_chunk_results = uow.image_chunk_results.get_by_query_and_pipeline(query_ids, pipeline_id)

            # Group chunk results by query_id
            chunk_results_by_query: dict[int | str, list[Any]] = {qid: [] for qid in query_ids}
            for r in chunk_results:
                if r.query_id in chunk_results_by_query:
                    chunk_results_by_query[r.query_id].append(r)

            image_chunk_results_by_query: dict[int | str, list[Any]] = {qid: [] for qid in query_ids}
            for r in image_chunk_results:
                if r.query_id in image_chunk_results_by_query:
                    image_chunk_results_by_query[r.query_id].append(r)

            for query_id in query_ids:
                chunk_results = chunk_results_by_query[query_id]
                chunk_ids = [(r.rel_score or 0.0, f"chunk_{r.chunk_id}") for r in chunk_results]
                image_chunk_results = image_chunk_results_by_query[query_id]
                image_chunk_ids = [(r.rel_score or 0.0, f"image_chunk_{r.image_chunk_id}") for r in image_chunk_results]

                all_results = chunk_ids + image_chunk_ids
                all_results.sort(key=lambda x: x[0], reverse=True)
                retrieved_ids = [prefixed_id for _, prefixed_id in all_results]

                # Get ground truth as 2D list (AND/OR structure) with prefixes and relevance scores
                gt_relations = uow.retrieval_relations.get_by_query_id(query_id)
                retrieval_gt, relevance_scores = build_retrieval_gt_from_relations(gt_relations)

                result[query_id] = {
                    "retrieved_ids": retrieved_ids,
                    "retrieval_gt": retrieval_gt,
                    "relevance_scores": relevance_scores,
                }

            return result

    def _filter_missing_query_ids(
        self, pipeline_id: int | str, metric_id: int | str, query_ids: list[int | str]
    ) -> list[int | str]:
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

    def _prepare_metric_input(
        self, pipeline_id: int | str, query_id: int | str, execution_result: dict[str, Any]
    ) -> MetricInput:
        """Prepare MetricInput for metric computation.

        Args:
            pipeline_id: The pipeline ID.
            query_id: The query ID.
            execution_result: Dict with 'retrieved_ids', 'retrieval_gt', and 'relevance_scores'.

        Returns:
            MetricInput instance ready for metric function.
        """
        return MetricInput(
            retrieved_ids=execution_result.get("retrieved_ids"),
            retrieval_gt=execution_result.get("retrieval_gt"),
            relevance_scores=execution_result.get("relevance_scores"),
        )

    def _has_results_for_queries(self, pipeline_id: int | str, query_ids: list[int | str]) -> bool:
        """Check if all given query IDs have retrieval results for the pipeline.

        Checks both ChunkRetrievedResult and ImageChunkRetrievedResult tables.

        Args:
            pipeline_id: The pipeline ID.
            query_ids: List of query IDs to check.

        Returns:
            True if all query IDs have results, False otherwise.
        """
        with self._create_uow() as uow:
            chunk_results = uow.chunk_results.get_by_query_and_pipeline(query_ids, pipeline_id)
            image_chunk_results = uow.image_chunk_results.get_by_query_and_pipeline(query_ids, pipeline_id)
            composite_dicts = {qid: [] for qid in query_ids}
            for r in chunk_results:
                if r.query_id is not None:
                    composite_dicts[r.query_id].append(r.chunk_id)
            for r in image_chunk_results:
                if r.query_id is not None:
                    composite_dicts[r.query_id].append(r.image_chunk_id)

            return all(composite_dicts[query_id] for query_id in query_ids)
