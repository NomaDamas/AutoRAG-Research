"""Base configuration classes for AutoRAG-Research.

This module provides base configuration dataclasses for pipelines, metrics,
and the Executor. Concrete implementations should be defined alongside their
respective pipeline/metric implementations.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from autorag_research.pipelines.retrieval.base import BaseRetrievalPipeline


class PipelineType(Enum):
    """Type of pipeline."""

    RETRIEVAL = "retrieval"
    GENERATION = "generation"


class MetricType(Enum):
    """Type of metric."""

    RETRIEVAL = "retrieval"
    GENERATION = "generation"


@dataclass
class BasePipelineConfig(ABC):
    """Base configuration for all pipelines.

    Subclasses should define their specific configuration parameters as dataclass
    fields and implement the abstract methods.

    Attributes:
        name: Unique name for this pipeline instance.
        pipeline_type: Type of pipeline (RETRIEVAL or GENERATION).
        top_k: Number of results to retrieve per query.
        batch_size: Number of queries to process in each batch.

    Example:
        ```python
        @dataclass
        class BM25PipelineConfig(BasePipelineConfig):
            index_path: str
            k1: float = 0.9
            b: float = 0.4
            pipeline_type: PipelineType = field(default=PipelineType.RETRIEVAL, init=False)

            def get_pipeline_class(self) -> Type:
                from autorag_research.pipelines.retrieval.bm25 import BM25RetrievalPipeline
                return BM25RetrievalPipeline

            def get_pipeline_kwargs(self) -> dict[str, Any]:
                return {"index_path": self.index_path, "k1": self.k1, "b": self.b}
        ```
    """

    name: str
    description: str = ""
    pipeline_type: PipelineType = field(init=False)
    top_k: int = 10
    batch_size: int = 100

    @abstractmethod
    def get_pipeline_class(self) -> type["BaseRetrievalPipeline"]:
        """Return the pipeline class to instantiate.

        Returns:
            The pipeline class type.
        """
        ...

    @abstractmethod
    def get_pipeline_kwargs(self) -> dict[str, Any]:
        """Return kwargs for pipeline constructor.

        These kwargs are passed to the pipeline constructor along with
        session_factory, name, and schema (which are handled by Executor).

        Returns:
            Dictionary of keyword arguments for the pipeline constructor.
        """
        ...

    @abstractmethod
    def get_run_kwargs(self) -> dict[str, Any]:
        """Return kwargs for pipeline.run() method.

        Returns:
            Dictionary of keyword arguments for the run method.
        """
        ...


@dataclass
class BaseRetrievalPipelineConfig(BasePipelineConfig, ABC):
    """Base configuration for retrieval pipelines.

    This class sets the pipeline_type to RETRIEVAL by default.
    """

    pipeline_type: PipelineType = field(default=PipelineType.RETRIEVAL, init=False)

    def get_run_kwargs(self) -> dict[str, Any]:
        """Return kwargs for pipeline.run() method.

        Returns:
            Dictionary of keyword arguments for the run method.
        """
        ...
        return {"top_k": self.top_k, "batch_size": self.batch_size}


@dataclass
class BaseGenerationPipelineConfig(BasePipelineConfig, ABC):
    """Base configuration for generation pipelines.

    This class sets the pipeline_type to GENERATION by default.
    """

    pipeline_type: PipelineType = field(default=PipelineType.GENERATION, init=False)
    retrieval_pipeline_name: str = ""
    # Runtime injection (Executor sets this)
    _retrieval_pipeline: "BaseRetrievalPipeline | None" = field(default=None, repr=False)

    def get_run_kwargs(self) -> dict[str, Any]:
        """Return kwargs for pipeline.run() method.

        Returns:
            Dictionary of keyword arguments for the run method.
        """
        return {"top_k": self.top_k, "batch_size": self.batch_size}

    def inject_retrieval_pipeline(self, pipeline: "BaseRetrievalPipeline") -> None:
        """Inject the retrieval pipeline instance.

        Called by Executor after loading and instantiating the retrieval pipeline.

        Args:
            pipeline: The instantiated retrieval pipeline.
        """
        self._retrieval_pipeline = pipeline


@dataclass
class BaseMetricConfig(ABC):
    """Base configuration for all metrics.

    Subclasses should define their specific configuration parameters as dataclass
    fields and implement the abstract methods.

    Attributes:
        metric_type: Type of metric (RETRIEVAL or GENERATION).

    Example:
        ```python
        @dataclass
        class RecallConfig(BaseMetricConfig):
            metric_type: MetricType = field(default=MetricType.RETRIEVAL, init=False)

            def get_metric_name(self) -> str:
                return "retrieval_recall"

            def get_metric_func(self) -> Callable:
                from autorag_research.evaluation.metrics import retrieval_recall
                return retrieval_recall
        ```
    """

    description: str = ""
    metric_type: MetricType = field(init=False)

    def get_metric_name(self) -> str:
        """Return the metric name for database storage.

        Returns:
            The metric name string.
        """
        return self.get_metric_func().__name__  # ty: ignore

    @abstractmethod
    def get_metric_func(self) -> Callable:
        """Return the metric function.

        Returns:
            The callable metric function.
        """
        ...

    def get_metric_kwargs(self) -> dict[str, Any]:
        """Return optional kwargs for the metric function.

        Override this method if the metric function accepts additional arguments.

        Returns:
            Dictionary of keyword arguments for the metric function.
        """
        return {}


@dataclass
class BaseRetrievalMetricConfig(BaseMetricConfig, ABC):
    """Base configuration for retrieval metrics.

    This class sets the metric_type to RETRIEVAL by default.
    """

    metric_type: MetricType = field(default=MetricType.RETRIEVAL, init=False)


@dataclass
class BaseGenerationMetricConfig(BaseMetricConfig, ABC):
    """Base configuration for generation metrics.

    This class sets the metric_type to GENERATION by default.
    """

    metric_type: MetricType = field(default=MetricType.GENERATION, init=False)


@dataclass
class ExecutorConfig:
    """Configuration for the Executor.

    Attributes:
        pipelines: List of pipeline configurations to run.
        metrics: List of metric configurations to evaluate.
        max_retries: Maximum number of retry attempts for failed pipelines.
        eval_batch_size: Batch size for metric evaluation.

    Example:
        ```python
        config = ExecutorConfig(
            pipelines=[
                BM25PipelineConfig(name="bm25_v1", index_path="/data/index"),
            ],
            metrics=[
                RecallConfig(),
                NDCGConfig(),
            ],
            max_retries=3,
        )
        ```
    """

    pipelines: list[BasePipelineConfig]
    metrics: list[BaseMetricConfig]
    max_retries: int = 3
    eval_batch_size: int = 100
