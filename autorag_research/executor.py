"""Executor for AutoRAG-Research.

This module provides the Executor class that orchestrates pipeline execution
and metric evaluation with retry logic, completion verification, and logging.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from autorag_research.cli.config_resolver import ConfigResolver
from autorag_research.cli.utils import get_config_dir
from autorag_research.config import (
    BaseMetricConfig,
    BasePipelineConfig,
    ExecutorConfig,
    MetricType,
    PipelineType,
)
from autorag_research.orm.service.generation_evaluation import GenerationEvaluationService
from autorag_research.orm.service.retrieval_evaluation import RetrievalEvaluationService
from autorag_research.pipelines.retrieval.base import BaseRetrievalPipeline

logger = logging.getLogger("AutoRAG-Research")


PIPELINE_TYPES = ["pipelines", "retrieval"]


@dataclass
class PipelineResult:
    """Result of a single pipeline execution.

    Attributes:
        pipeline_id: The database ID of the pipeline.
        pipeline_name: The name of the pipeline.
        pipeline_type: The type of pipeline (RETRIEVAL or GENERATION).
        total_queries: Total number of queries processed.
        total_results: Total number of results stored.
        retries_used: Number of retry attempts used.
        success: Whether the pipeline execution was successful.
        error_message: Error message if the pipeline failed.
    """

    pipeline_id: int | str
    pipeline_name: str
    pipeline_type: PipelineType
    total_queries: int
    total_results: int
    retries_used: int
    success: bool
    error_message: str | None = None


@dataclass
class MetricResult:
    """Result of a single metric evaluation.

    Attributes:
        metric_name: The name of the metric.
        metric_type: The type of metric (RETRIEVAL or GENERATION).
        pipeline_id: The pipeline ID that was evaluated.
        queries_evaluated: Number of queries evaluated.
        average: Average metric score across all evaluated queries.
        success: Whether the evaluation was successful.
        error_message: Error message if the evaluation failed.
    """

    metric_name: str
    metric_type: MetricType
    pipeline_id: int | str
    queries_evaluated: int
    average: float | None
    success: bool
    error_message: str | None = None


@dataclass
class ExecutorResult:
    """Complete result of Executor run.

    Attributes:
        pipeline_results: Results for each pipeline execution.
        metric_results: Results for each metric evaluation.
        total_pipelines_run: Total number of pipelines run.
        total_pipelines_succeeded: Number of pipelines that succeeded.
        total_metrics_evaluated: Total number of metric evaluations.
        total_metrics_succeeded: Number of metric evaluations that succeeded.
    """

    pipeline_results: list[PipelineResult] = field(default_factory=list)
    metric_results: list[MetricResult] = field(default_factory=list)
    total_pipelines_run: int = 0
    total_pipelines_succeeded: int = 0
    total_metrics_evaluated: int = 0
    total_metrics_succeeded: int = 0


class Executor:
    """Orchestrates pipeline execution and metric evaluation.

    The Executor coordinates:
    1. Sequential execution of configured pipelines
    2. Verification that all queries have results
    3. Retry logic for failed pipelines
    4. Metric evaluation for each pipeline (immediately after pipeline completes)

    Metric Evaluation Rules:
    - Retrieval pipelines: Only retrieval metrics are evaluated
    - Generation pipelines: Both retrieval AND generation metrics are evaluated

    Example:
        ```python
        from autorag_research.config import ExecutorConfig
        from autorag_research.executor import Executor
        from autorag_research.orm.connection import DBConnection
        from autorag_research.pipelines.retrieval.bm25 import BM25PipelineConfig
        from autorag_research.evaluation.metrics.retrieval import RecallConfig, NDCGConfig

        db = DBConnection.from_config()  # or DBConnection.from_env()
        session_factory = db.get_session_factory()

        config = ExecutorConfig(
            pipelines=[
                BM25PipelineConfig(
                    name="bm25_baseline",
                    tokenizer="bert",
                    top_k=10,
                ),
            ],
            metrics=[
                RecallConfig(),
                NDCGConfig(),
            ],
            max_retries=3,
        )

        executor = Executor(session_factory, config)
        result = executor.run()
        ```
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        config: ExecutorConfig,
        schema: Any | None = None,
        config_dir: Path | None = None,
    ):
        """Initialize Executor.

        Args:
            session_factory: SQLAlchemy sessionmaker for database connections.
            config: Executor configuration.
            schema: Schema namespace from create_schema(). If None, uses default schema.
            config_dir: Directory containing pipeline YAML configs. If None, attempts to
                use Hydra's config path if initialized, otherwise falls back to CWD/configs.
        """
        self.session_factory = session_factory
        self.config = config
        self._schema = schema
        self._config_dir = config_dir or get_config_dir()
        self._config_resolver = ConfigResolver(self._config_dir)

        # Initialize evaluation services
        self._retrieval_eval_service = RetrievalEvaluationService(session_factory, schema)
        self._generation_eval_service = GenerationEvaluationService(session_factory, schema)

        # Cache for dependency retrieval pipelines (loaded from YAML)
        self._dependency_pipelines: dict[str, BaseRetrievalPipeline] = {}

        logger.info(f"Executor initialized with {len(config.pipelines)} pipelines and {len(config.metrics)} metrics")
        logger.debug(f"Config directory: {self._config_dir}")

    def run(self) -> ExecutorResult:
        """Run all configured pipelines and evaluate metrics.

        For each pipeline:
        1. Run the pipeline with retry logic
        2. Verify completion
        3. Evaluate applicable metrics (before moving to next pipeline)

        Returns:
            ExecutorResult with comprehensive execution statistics.
        """
        result = ExecutorResult()

        for pipeline_config in self.config.pipelines:
            logger.info(f"=== Starting pipeline: {pipeline_config.name} ===")

            # Step 0: Resolve dependencies for generation pipelines
            if pipeline_config.pipeline_type == PipelineType.GENERATION:
                self._resolve_dependencies(pipeline_config)

            # Step 1: Run pipeline with retry
            pipeline_result = self._run_pipeline_with_retry(pipeline_config)
            result.pipeline_results.append(pipeline_result)
            result.total_pipelines_run += 1

            if pipeline_result.success:
                result.total_pipelines_succeeded += 1

                # Step 2: Evaluate metrics for this pipeline (before next pipeline)
                metric_results = self._evaluate_metrics_for_pipeline(
                    pipeline_result.pipeline_id,
                    pipeline_config.pipeline_type,
                )
                result.metric_results.extend(metric_results)
                result.total_metrics_evaluated += len(metric_results)
                result.total_metrics_succeeded += sum(1 for m in metric_results if m.success)

            logger.info(f"=== Completed pipeline: {pipeline_config.name} (success={pipeline_result.success}) ===")

        logger.info(
            f"Executor complete: "
            f"{result.total_pipelines_succeeded}/{result.total_pipelines_run} pipelines, "
            f"{result.total_metrics_succeeded}/{result.total_metrics_evaluated} metrics"
        )

        return result

    def _run_pipeline_with_retry(self, config: BasePipelineConfig) -> PipelineResult:
        """Run a single pipeline with retry logic.

        Args:
            config: Pipeline configuration.

        Returns:
            PipelineResult with execution details.
        """
        logger.info(f"Starting pipeline: {config.name} ({config.pipeline_type.value})")
        error_msg = ""

        for attempt in range(self.config.max_retries + 1):
            if attempt > 0:
                logger.warning(f"Retry {attempt}/{self.config.max_retries} for pipeline: {config.name}")

            try:
                # Instantiate pipeline
                pipeline_class = config.get_pipeline_class()
                pipeline = pipeline_class(
                    session_factory=self.session_factory,
                    name=config.name,
                    schema=self._schema,
                    **config.get_pipeline_kwargs(),
                )

                # Run pipeline
                run_result = pipeline.run(**config.get_run_kwargs())
                pipeline_id = run_result["pipeline_id"]

                # Verify completion
                if self._verify_pipeline_completion(pipeline_id, config.pipeline_type):
                    logger.info(
                        f"Pipeline '{config.name}' completed successfully "
                        f"(pipeline_id={pipeline_id}, "
                        f"queries={run_result['total_queries']}, "
                        f"results={run_result['total_results']})"
                    )

                    return PipelineResult(
                        pipeline_id=pipeline_id,
                        pipeline_name=config.name,
                        pipeline_type=config.pipeline_type,
                        total_queries=run_result["total_queries"],
                        total_results=run_result["total_results"],
                        retries_used=attempt,
                        success=True,
                    )
                else:
                    logger.warning(f"Pipeline '{config.name}' verification failed, will retry")

            except Exception as e:
                error_msg += f"\n\n{e}"
                logger.exception(f"Pipeline '{config.name}' failed with error")

        # All retries exhausted
        logger.error(f"Pipeline '{config.name}' failed: {error_msg}")

        return PipelineResult(
            pipeline_id=-1,
            pipeline_name=config.name,
            pipeline_type=config.pipeline_type,
            total_queries=0,
            total_results=0,
            retries_used=self.config.max_retries,
            success=False,
            error_message=error_msg,
        )

    def _verify_pipeline_completion(self, pipeline_id: int | str, pipeline_type: PipelineType) -> bool:
        """Verify all queries have results for the pipeline.

        Uses ID comparison to check that each query has a corresponding result.
        Delegates to appropriate evaluation service based on pipeline type.

        Args:
            pipeline_id: The pipeline ID to verify.
            pipeline_type: Type of pipeline (determines which service to use).

        Returns:
            True if all queries have results, False otherwise.
        """
        if pipeline_type == PipelineType.RETRIEVAL:
            return self._retrieval_eval_service.verify_pipeline_completion(pipeline_id)
        else:
            return self._generation_eval_service.verify_pipeline_completion(pipeline_id)

    def _evaluate_metrics_for_pipeline(self, pipeline_id: int | str, pipeline_type: PipelineType) -> list[MetricResult]:
        """Evaluate all applicable metrics for a pipeline.

        Metric evaluation rules:
        - Retrieval pipelines: Only retrieval metrics
        - Generation pipelines: Both retrieval AND generation metrics

        Args:
            pipeline_id: The pipeline ID to evaluate.
            pipeline_type: Type of pipeline.

        Returns:
            List of MetricResult for each metric evaluated.
        """
        results = []

        for metric_config in self.config.metrics:
            # Apply metric evaluation rules
            # Retrieval pipeline: only retrieval metrics
            if pipeline_type == PipelineType.RETRIEVAL and metric_config.metric_type == MetricType.GENERATION:
                logger.debug(
                    f"Skipping generation metric '{metric_config.get_metric_name()}' "
                    f"for retrieval pipeline {pipeline_id}"
                )
                continue
            # Generation pipeline: evaluate ALL metrics (both retrieval and generation)

            metric_result = self._evaluate_metric(pipeline_id, metric_config)
            results.append(metric_result)

        return results

    def _evaluate_metric(self, pipeline_id: int | str, config: BaseMetricConfig) -> MetricResult:
        """Evaluate a single metric for a pipeline.

        Args:
            pipeline_id: The pipeline ID to evaluate.
            config: Metric configuration.

        Returns:
            MetricResult with evaluation details.
        """
        metric_name = config.get_metric_name()
        logger.info(f"Evaluating metric '{metric_name}' for pipeline_id={pipeline_id}")

        try:
            # Get metric function and kwargs
            metric_func = config.get_metric_func()
            metric_kwargs = config.get_metric_kwargs()

            # Select appropriate evaluation service
            if config.metric_type == MetricType.RETRIEVAL:
                service = self._retrieval_eval_service
            else:
                service = self._generation_eval_service

            # Get or create metric in DB
            metric_id = service.get_or_create_metric(
                name=metric_name,
                metric_type=config.metric_type.value,
            )

            # Set metric function that accepts list[MetricInput] and returns list[float | None]
            service.set_metric(
                metric_id=metric_id,
                metric_func=lambda metric_inputs: metric_func(metric_inputs, **metric_kwargs),
            )

            # Evaluate
            queries_evaluated, average = service.evaluate(
                pipeline_id=pipeline_id,
                batch_size=self.config.eval_batch_size,
            )

            logger.info(
                f"Metric '{metric_name}' evaluated: {queries_evaluated} queries for pipeline_id={pipeline_id}, "
                f"average={average}"
            )

            return MetricResult(
                metric_name=metric_name,
                metric_type=config.metric_type,
                pipeline_id=pipeline_id,
                queries_evaluated=queries_evaluated,
                average=average,
                success=True,
            )

        except Exception as e:
            error_msg = str(e)
            logger.exception(f"Metric '{metric_name}' evaluation failed for pipeline_id={pipeline_id}")
            return MetricResult(
                metric_name=metric_name,
                metric_type=config.metric_type,
                pipeline_id=pipeline_id,
                queries_evaluated=0,
                average=None,
                success=False,
                error_message=error_msg,
            )

    def _resolve_dependencies(self, config: BasePipelineConfig) -> None:
        """Resolve dependencies for a generation pipeline config.

        Checks for `retrieval_pipeline_name` attribute and loads/instantiates
        the referenced retrieval pipeline, then injects it into the config.

        Uses Hydra's compose API when available to leverage configured search paths,
        with fallback to direct YAML loading from config_dir.

        Args:
            config: Pipeline configuration (processes any BaseGenerationPipelineConfig
                with a non-empty retrieval_pipeline_name).
        """
        from hydra.utils import instantiate

        from autorag_research.config import BaseGenerationPipelineConfig

        # Only process generation pipeline configs
        if not isinstance(config, BaseGenerationPipelineConfig):
            return

        # Skip if retrieval pipeline already injected (programmatic usage)
        if config._retrieval_pipeline is not None:
            logger.debug(f"Retrieval pipeline already injected for {config.name}")
            return

        # Skip if no retrieval pipeline dependency
        name: str = config.retrieval_pipeline_name
        if not name:
            return

        logger.info(f"Resolving retrieval pipeline dependency: {name}")

        # Return cached pipeline if already loaded
        if name in self._dependency_pipelines:
            logger.debug(f"Using cached retrieval pipeline: {name}")
            config.inject_retrieval_pipeline(self._dependency_pipelines[name])
            return

        # Load pipeline config from pipelines/retrieval/ directory
        pipeline_cfg = self._config_resolver.resolve_config(PIPELINE_TYPES, name)
        pipeline_config = instantiate(pipeline_cfg)

        # Instantiate the retrieval pipeline
        pipeline_class = pipeline_config.get_pipeline_class()
        retrieval_pipeline = pipeline_class(
            session_factory=self.session_factory,
            name=pipeline_config.name,
            schema=self._schema,
            **pipeline_config.get_pipeline_kwargs(),
        )

        # Cache and inject
        self._dependency_pipelines[name] = retrieval_pipeline
        config.inject_retrieval_pipeline(retrieval_pipeline)
        logger.info(f"Loaded and injected retrieval pipeline: {name}")
