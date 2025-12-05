"""Evaluation Unit of Work for managing retrieval evaluation operations.

Provides atomic transaction management for retrieval evaluation including:
- Query fetching
- Pipeline and Metric configuration
- RetrievalRelation for ground truth
- ChunkRetrievedResult and ImageChunkRetrievedResult for retrieval outputs
- EvaluationResult for storing evaluation scores

Generation Unit of Work for managing generation pipeline transactions.

Provides atomic transaction management for generation operations including:
- Query fetching
- Pipeline configuration
- Executor results (generation outputs)
- Evaluation results (metric scores)
- Metric definitions
"""

from typing import Any

from sqlalchemy.orm import sessionmaker

from autorag_research.orm.repository import ExecutorResultRepository
from autorag_research.orm.repository.chunk_retrieved_result import (
    ChunkRetrievedResultRepository,
)
from autorag_research.orm.repository.evaluator_result import EvaluatorResultRepository
from autorag_research.orm.repository.metric import MetricRepository
from autorag_research.orm.repository.pipeline import PipelineRepository
from autorag_research.orm.repository.query import QueryRepository
from autorag_research.orm.repository.retrieval_relation import (
    RetrievalRelationRepository,
)
from autorag_research.orm.uow.base import BaseUnitOfWork


class RetrievalEvaluationUnitOfWork(BaseUnitOfWork):
    """Unit of Work for retrieval evaluation operations.

    Manages transactions across multiple repositories needed for evaluation:
    - Query: For fetching queries to evaluate
    - Pipeline: For pipeline configuration and tracking
    - Metric: For metric definitions and tracking
    - RetrievalRelation: For ground truth relevance data
    - ChunkRetrievedResult: For text retrieval results to evaluate
    - ImageChunkRetrievedResult: For image retrieval results to evaluate
    - EvaluationResult: For storing computed evaluation scores

    Example:
        ```python
        with RetrievalEvaluationUnitOfWork(session_factory) as uow:
            # Fetch queries and ground truth
            queries = uow.queries.get_all()
            gt_relations = uow.retrieval_relations.get_by_query_id(query_id)

            # Get retrieval results
            retrieval_results = uow.chunk_results.get_by_query(query_id)

            # Compute metrics and save
            metric = uow.metrics.get_by_name("retrieval@k")
            score = compute_metric(retrieval_results, gt_relations)

            eval_result = EvaluationResult(
                query_id=query_id,
                pipeline_id=pipeline_id,
                metric_id=metric.id,
                score=score
            )
            uow.evaluation_results.add(eval_result)
            uow.commit()
        ```
    """

    def __init__(self, session_factory: sessionmaker, schema: Any | None = None):
        """Initialize RetrievalEvaluationUnitOfWork with session factory and schema.

        Args:
            session_factory: SQLAlchemy sessionmaker instance.
            schema: Schema namespace from create_schema(). If None, uses default 768-dim schema.
        """
        super().__init__(session_factory, schema)

        # Lazy-initialized repositories
        self._query_repo: QueryRepository | None = None
        self._pipeline_repo: PipelineRepository | None = None
        self._metric_repo: MetricRepository | None = None
        self._retrieval_relation_repo: RetrievalRelationRepository | None = None
        self._chunk_result_repo: ChunkRetrievedResultRepository | None = None
        self._evaluation_result_repo: EvaluatorResultRepository | None = None

    def _get_schema_classes(self) -> dict[str, type]:
        """Get all model classes from schema.

        Returns:
            Dictionary mapping class names to model classes.
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

    def _reset_repositories(self) -> None:
        """Reset all repository references to None."""
        self._query_repo = None
        self._pipeline_repo = None
        self._metric_repo = None
        self._retrieval_relation_repo = None
        self._chunk_result_repo = None
        self._evaluation_result_repo = None

    @property
    def queries(self) -> QueryRepository:
        """Get the Query repository.

        Returns:
            QueryRepository instance.

        Raises:
            SessionNotSetError: If session is not initialized.
        """
        return self._get_repository(
            "_query_repo",
            QueryRepository,
            lambda: self._get_schema_classes()["Query"],
        )

    @property
    def pipelines(self) -> PipelineRepository:
        """Get the Pipeline repository.

        Returns:
            PipelineRepository instance.

        Raises:
            SessionNotSetError: If session is not initialized.
        """
        return self._get_repository(
            "_pipeline_repo",
            PipelineRepository,
            lambda: self._get_schema_classes()["Pipeline"],
        )

    @property
    def metrics(self) -> MetricRepository:
        """Get the Metric repository.

        Returns:
            MetricRepository instance.

        Raises:
            SessionNotSetError: If session is not initialized.
        """
        return self._get_repository(
            "_metric_repo",
            MetricRepository,
            lambda: self._get_schema_classes()["Metric"],
        )

    @property
    def retrieval_relations(self) -> RetrievalRelationRepository:
        """Get the RetrievalRelation repository.

        Returns:
            RetrievalRelationRepository instance.

        Raises:
            SessionNotSetError: If session is not initialized.
        """
        return self._get_repository(
            "_retrieval_relation_repo",
            RetrievalRelationRepository,
            lambda: self._get_schema_classes()["RetrievalRelation"],
        )

    @property
    def chunk_results(self) -> ChunkRetrievedResultRepository:
        """Get the ChunkRetrievedResult repository.

        Returns:
            ChunkRetrievedResultRepository instance.

        Raises:
            SessionNotSetError: If session is not initialized.
        """
        return self._get_repository(
            "_chunk_result_repo",
            ChunkRetrievedResultRepository,
            lambda: self._get_schema_classes()["ChunkRetrievedResult"],
        )

    @property
    def evaluation_results(self) -> EvaluatorResultRepository:
        """Get the EvaluationResult repository.

        Returns:
            EvaluatorResultRepository instance.

        Raises:
            SessionNotSetError: If session is not initialized.
        """
        return self._get_repository(
            "_evaluation_result_repo",
            EvaluatorResultRepository,
            lambda: self._get_schema_classes()["EvaluationResult"],
        )


class GenerationEvaluationUnitOfWork(BaseUnitOfWork):
    """Unit of Work for generation pipeline operations.

    Manages transactions across multiple repositories needed for generation:
    - Query: For fetching evaluation queries
    - Pipeline: For configuration and tracking
    - ExecutorResult: For storing generation outputs
    - EvaluationResult: For storing metric evaluation scores
    - Metric: For metric definitions
    """

    def __init__(self, session_factory: sessionmaker, schema: Any | None = None):
        """Initialize GenerationUnitOfWork with session factory and schema.

        Args:
            session_factory: SQLAlchemy sessionmaker instance.
            schema: Schema namespace from create_schema(). If None, uses default 768-dim schema.
        """
        super().__init__(session_factory, schema)

        # Lazy-initialized repositories
        self._query_repo: QueryRepository | None = None
        self._pipeline_repo: PipelineRepository | None = None
        self._executor_result_repo: ExecutorResultRepository | None = None
        self._evaluator_result_repo: EvaluatorResultRepository | None = None
        self._metric_repo: MetricRepository | None = None

    def _get_schema_classes(self) -> dict[str, type]:
        """Get all model classes from schema.

        Returns:
            Dictionary mapping class names to model classes.
        """
        if self._schema is not None:
            return {
                "Query": self._schema.Query,
                "Pipeline": self._schema.Pipeline,
                "ExecutorResult": self._schema.ExecutorResult,
                "EvaluationResult": self._schema.EvaluationResult,
                "Metric": self._schema.Metric,
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
            "ExecutorResult": ExecutorResult,
            "EvaluationResult": EvaluationResult,
            "Metric": Metric,
        }

    def _reset_repositories(self) -> None:
        """Reset all repository references to None."""
        self._query_repo = None
        self._pipeline_repo = None
        self._executor_result_repo = None
        self._evaluator_result_repo = None
        self._metric_repo = None

    @property
    def queries(self) -> QueryRepository:
        """Get the Query repository.

        Returns:
            QueryRepository instance.

        Raises:
            SessionNotSetError: If session is not initialized.
        """
        return self._get_repository(
            "_query_repo",
            QueryRepository,
            lambda: self._get_schema_classes()["Query"],
        )

    @property
    def pipelines(self) -> PipelineRepository:
        """Get the Pipeline repository.

        Returns:
            PipelineRepository instance.

        Raises:
            SessionNotSetError: If session is not initialized.
        """
        return self._get_repository(
            "_pipeline_repo",
            PipelineRepository,
            lambda: self._get_schema_classes()["Pipeline"],
        )

    @property
    def executor_results(self) -> ExecutorResultRepository:
        """Get the ExecutorResult repository.

        Returns:
            ExecutorResultRepository instance.

        Raises:
            SessionNotSetError: If session is not initialized.
        """
        return self._get_repository(
            "_executor_result_repo",
            ExecutorResultRepository,
            lambda: self._get_schema_classes()["ExecutorResult"],
        )

    @property
    def evaluation_results(self) -> EvaluatorResultRepository:
        """Get the EvaluationResult repository.

        Returns:
            EvaluatorResultRepository instance.

        Raises:
            SessionNotSetError: If session is not initialized.
        """
        return self._get_repository(
            "_evaluator_result_repo",
            EvaluatorResultRepository,
            lambda: self._get_schema_classes()["EvaluationResult"],
        )

    @property
    def metrics(self) -> MetricRepository:
        """Get the Metric repository.

        Returns:
            MetricRepository instance.

        Raises:
            SessionNotSetError: If session is not initialized.
        """
        return self._get_repository(
            "_metric_repo",
            MetricRepository,
            lambda: self._get_schema_classes()["Metric"],
        )
