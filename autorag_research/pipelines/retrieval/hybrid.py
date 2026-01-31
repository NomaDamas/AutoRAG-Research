"""Hybrid Retrieval Pipelines for AutoRAG-Research.

This module provides hybrid retrieval pipelines that combine results from
two retrieval methods using different fusion strategies:
- RRF (Reciprocal Rank Fusion): Rank-based fusion, ignores scores
- CC (Convex Combination): Score-based fusion with normalization
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from sqlalchemy.orm import Session, sessionmaker

from autorag_research.config import BaseRetrievalPipelineConfig
from autorag_research.pipelines.retrieval.base import BaseRetrievalPipeline
from autorag_research.util import (
    normalize_dbsf,
    normalize_minmax,
    normalize_tmm,
    normalize_zscore,
)

NormalizationMethod = Literal["mm", "tmm", "z", "dbsf"]


def _rrf_fuse(
    results_1: list[dict[str, Any]],
    results_2: list[dict[str, Any]],
    k: int,
    top_k: int,
) -> list[dict[str, Any]]:
    """Fuse two result lists using Reciprocal Rank Fusion.

    RRF(d) = Σ 1/(k + rank_i(d)) for each result list i

    Args:
        results_1: First result list with 'doc_id' and 'score' keys.
        results_2: Second result list with 'doc_id' and 'score' keys.
        k: RRF constant (typically 60). Higher values give more weight to top ranks.
        top_k: Number of results to return.

    Returns:
        Fused results sorted by RRF score (descending).
    """
    rrf_scores: dict[int, float] = {}

    # Process first result list
    for rank, result in enumerate(results_1, start=1):
        doc_id = result["doc_id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    # Process second result list
    for rank, result in enumerate(results_2, start=1):
        doc_id = result["doc_id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    # Sort by RRF score and return top_k
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [{"doc_id": doc_id, "score": score} for doc_id, score in sorted_docs[:top_k]]


def _cc_fuse(
    results_1: list[dict[str, Any]],
    results_2: list[dict[str, Any]],
    weight: float,
    top_k: int,
    normalize_method: NormalizationMethod,
    pipeline_1_min: float | None = None,
    pipeline_2_min: float | None = None,
) -> list[dict[str, Any]]:
    """Fuse two result lists using Convex Combination with score normalization.

    combined = weight x norm(scores_1) + (1-weight) x norm(scores_2)

    Args:
        results_1: First result list with 'doc_id' and 'score' keys.
        results_2: Second result list with 'doc_id' and 'score' keys.
        weight: Weight for pipeline_1 scores (0-1). 0 = full pipeline_2, 1 = full pipeline_1.
        top_k: Number of results to return.
        normalize_method: Normalization method ("mm", "tmm", "z", "dbsf").
        pipeline_1_min: Theoretical min for TMM normalization (pipeline_1).
        pipeline_2_min: Theoretical min for TMM normalization (pipeline_2).

    Returns:
        Fused results sorted by combined score (descending).
    """
    # Extract doc_ids and scores
    doc_ids_1 = [r["doc_id"] for r in results_1]
    scores_1 = [r["score"] for r in results_1]
    doc_ids_2 = [r["doc_id"] for r in results_2]
    scores_2 = [r["score"] for r in results_2]

    # Normalize scores
    if normalize_method == "mm":
        norm_scores_1 = normalize_minmax(scores_1)
        norm_scores_2 = normalize_minmax(scores_2)
    elif normalize_method == "tmm":
        if pipeline_1_min is None:
            msg = "TMM normalization requires pipeline_1_min"
            raise ValueError(msg)
        if pipeline_2_min is None:
            msg = "TMM normalization requires pipeline_2_min"
            raise ValueError(msg)
        norm_scores_1 = normalize_tmm(scores_1, pipeline_1_min)
        norm_scores_2 = normalize_tmm(scores_2, pipeline_2_min)
    elif normalize_method == "z":
        norm_scores_1 = normalize_zscore(scores_1)
        norm_scores_2 = normalize_zscore(scores_2)
    elif normalize_method == "dbsf":
        norm_scores_1 = normalize_dbsf(scores_1)
        norm_scores_2 = normalize_dbsf(scores_2)
    else:
        msg = f"Unknown normalization method: {normalize_method}"
        raise ValueError(msg)

    # Build score maps
    score_map_1 = dict(zip(doc_ids_1, norm_scores_1, strict=True))
    score_map_2 = dict(zip(doc_ids_2, norm_scores_2, strict=True))

    # Combine scores for all unique documents
    all_doc_ids = set(doc_ids_1) | set(doc_ids_2)
    combined_scores: dict[int, float] = {}

    for doc_id in all_doc_ids:
        score_1 = score_map_1.get(doc_id, 0.0)
        score_2 = score_map_2.get(doc_id, 0.0)
        combined_scores[doc_id] = weight * score_1 + (1 - weight) * score_2

    # Sort by combined score and return top_k
    sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    return [{"doc_id": doc_id, "score": score} for doc_id, score in sorted_docs[:top_k]]


@dataclass(kw_only=True)
class HybridRetrievalPipelineConfig(BaseRetrievalPipelineConfig, ABC):
    """Base configuration for hybrid retrieval pipelines.

    Attributes:
        name: Unique name for this pipeline instance.
        retrieval_pipeline_1_name: Name of the first retrieval pipeline.
        retrieval_pipeline_2_name: Name of the second retrieval pipeline.
        top_k: Number of results to retrieve per query.
        batch_size: Number of queries to process in each batch.
    """

    retrieval_pipeline_1_name: str
    retrieval_pipeline_2_name: str


@dataclass(kw_only=True)
class HybridRRFRetrievalPipelineConfig(HybridRetrievalPipelineConfig):
    """Configuration for Hybrid RRF Retrieval Pipeline.

    Attributes:
        name: Unique name for this pipeline instance.
        retrieval_pipeline_1_name: Name of the first retrieval pipeline.
        retrieval_pipeline_2_name: Name of the second retrieval pipeline.
        rrf_k: RRF constant (default: 60). Higher values emphasize top ranks more.
        top_k: Number of results to retrieve per query.
        batch_size: Number of queries to process in each batch.

    Example:
        ```python
        config = HybridRRFRetrievalPipelineConfig(
            name="hybrid_rrf",
            retrieval_pipeline_1_name="vector_search",
            retrieval_pipeline_2_name="bm25",
            rrf_k=60,
            top_k=10,
        )
        ```
    """

    rrf_k: int = 60

    def get_pipeline_class(self) -> type["HybridRRFRetrievalPipeline"]:
        """Return the HybridRRFRetrievalPipeline class."""
        return HybridRRFRetrievalPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        """Return kwargs for HybridRRFRetrievalPipeline constructor."""
        return {
            "retrieval_pipeline_1": self.retrieval_pipeline_1_name,
            "retrieval_pipeline_2": self.retrieval_pipeline_2_name,
            "rrf_k": self.rrf_k,
        }


@dataclass(kw_only=True)
class HybridCCRetrievalPipelineConfig(HybridRetrievalPipelineConfig):
    """Configuration for Hybrid Convex Combination Retrieval Pipeline.

    Attributes:
        name: Unique name for this pipeline instance.
        retrieval_pipeline_1_name: Name of the first retrieval pipeline.
        retrieval_pipeline_2_name: Name of the second retrieval pipeline.
        weight: Weight for pipeline_1 scores (0-1). Default 0.5.
        normalize_method: Score normalization method ("mm", "tmm", "z", "dbsf").
        pipeline_1_min: Theoretical min score for TMM (pipeline_1).
        pipeline_2_min: Theoretical min score for TMM (pipeline_2).
        top_k: Number of results to retrieve per query.
        batch_size: Number of queries to process in each batch.

    Example:
        ```python
        config = HybridCCRetrievalPipelineConfig(
            name="hybrid_cc",
            retrieval_pipeline_1_name="vector_search",
            retrieval_pipeline_2_name="bm25",
            weight=0.5,
            normalize_method="mm",
            top_k=10,
        )
        ```
    """

    weight: float = 0.5
    normalize_method: NormalizationMethod = "mm"
    pipeline_1_min: float | None = None
    pipeline_2_min: float | None = None

    def get_pipeline_class(self) -> type["HybridCCRetrievalPipeline"]:
        """Return the HybridCCRetrievalPipeline class."""
        return HybridCCRetrievalPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        """Return kwargs for HybridCCRetrievalPipeline constructor."""
        return {
            "retrieval_pipeline_1": self.retrieval_pipeline_1_name,
            "retrieval_pipeline_2": self.retrieval_pipeline_2_name,
            "weight": self.weight,
            "normalize_method": self.normalize_method,
            "pipeline_1_min": self.pipeline_1_min,
            "pipeline_2_min": self.pipeline_2_min,
        }


class HybridRetrievalPipeline(BaseRetrievalPipeline, ABC):
    """Abstract base class for hybrid retrieval pipelines.

    This class provides common functionality for hybrid pipelines that
    combine results from two retrieval pipelines.

    Subclasses must implement:
    - `_fuse_results()`: Define the fusion strategy (RRF, CC, etc.)
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        retrieval_pipeline_1: "BaseRetrievalPipeline | str",
        retrieval_pipeline_2: "BaseRetrievalPipeline | str",
        schema: Any | None = None,
        config_dir: Path | None = None,
    ):
        """Initialize hybrid retrieval pipeline.

        Args:
            session_factory: SQLAlchemy sessionmaker for database connections.
            name: Name for this pipeline.
            retrieval_pipeline_1: First retrieval pipeline (instance or name string).
            retrieval_pipeline_2: Second retrieval pipeline (instance or name string).
            schema: Schema namespace from create_schema(). If None, uses default schema.
            config_dir: Directory containing pipeline configs (for loading by name).
        """
        # Load pipelines if strings provided
        if isinstance(retrieval_pipeline_1, str):
            retrieval_pipeline_1 = self._load_pipeline(retrieval_pipeline_1, session_factory, schema, config_dir)
        if isinstance(retrieval_pipeline_2, str):
            retrieval_pipeline_2 = self._load_pipeline(retrieval_pipeline_2, session_factory, schema, config_dir)

        # Store pipelines before super().__init__ (needed for _get_pipeline_config)
        self._retrieval_pipeline_1 = retrieval_pipeline_1
        self._retrieval_pipeline_2 = retrieval_pipeline_2

        super().__init__(session_factory, name, schema)

    @staticmethod
    def _load_pipeline(
        name: str,
        session_factory: sessionmaker[Session],
        schema: Any | None,
        config_dir: Path | None,
    ) -> BaseRetrievalPipeline:
        """Load a retrieval pipeline by name from YAML config.

        Args:
            name: Pipeline name (matches YAML filename without extension).
            session_factory: SQLAlchemy sessionmaker for database connections.
            schema: Schema namespace from create_schema().
            config_dir: Directory containing pipeline configs.

        Returns:
            Instantiated retrieval pipeline.
        """
        from hydra.utils import instantiate

        from autorag_research.cli.config_resolver import ConfigResolver
        from autorag_research.cli.utils import get_config_dir

        config_dir = config_dir or get_config_dir()
        resolver = ConfigResolver(config_dir)
        pipeline_cfg = resolver.resolve_config(["pipelines", "retrieval"], name)
        pipeline_config: BaseRetrievalPipelineConfig = instantiate(pipeline_cfg)

        pipeline_class = pipeline_config.get_pipeline_class()
        return pipeline_class(
            session_factory=session_factory,
            name=pipeline_config.name,
            schema=schema,
            **pipeline_config.get_pipeline_kwargs(),
        )

    @abstractmethod
    def _fuse_results(
        self,
        results_1: list[dict[str, Any]],
        results_2: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Fuse results from two pipelines.

        Args:
            results_1: Results from pipeline_1.
            results_2: Results from pipeline_2.
            top_k: Number of results to return.

        Returns:
            Fused results sorted by combined score.
        """
        pass

    def _get_retrieval_func(self) -> Any:
        """Return hybrid retrieval function."""
        func_1 = self._retrieval_pipeline_1._get_retrieval_func()
        func_2 = self._retrieval_pipeline_2._get_retrieval_func()

        def hybrid_retrieval(query_ids: list[int], top_k: int) -> list[list[dict]]:
            # Fetch more results from each pipeline for better fusion
            fetch_k = top_k * 2
            results_1 = func_1(query_ids, fetch_k)
            results_2 = func_2(query_ids, fetch_k)

            return [self._fuse_results(r1, r2, top_k) for r1, r2 in zip(results_1, results_2, strict=True)]

        return hybrid_retrieval


class HybridRRFRetrievalPipeline(HybridRetrievalPipeline):
    """Hybrid retrieval pipeline using Reciprocal Rank Fusion.

    RRF combines results based on rank positions, ignoring raw scores.
    This makes it robust to different score scales between pipelines.

    Formula: RRF(d) = Σ 1/(k + rank_i(d))

    Example:
        ```python
        from autorag_research.pipelines.retrieval import (
            HybridRRFRetrievalPipeline,
            VectorSearchRetrievalPipeline,
            BM25RetrievalPipeline,
        )

        # Create sub-pipelines
        vector = VectorSearchRetrievalPipeline(session_factory, "vector")
        bm25 = BM25RetrievalPipeline(session_factory, "bm25")

        # Create hybrid with instantiated pipelines
        hybrid = HybridRRFRetrievalPipeline(
            session_factory=session_factory,
            name="hybrid_rrf",
            retrieval_pipeline_1=vector,
            retrieval_pipeline_2=bm25,
            rrf_k=60,
        )

        # Or with pipeline names (auto-loaded from YAML configs)
        hybrid = HybridRRFRetrievalPipeline(
            session_factory=session_factory,
            name="hybrid_rrf",
            retrieval_pipeline_1="vector_search",
            retrieval_pipeline_2="bm25",
        )

        results = hybrid.retrieve("What is machine learning?", top_k=10)
        ```
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        retrieval_pipeline_1: "BaseRetrievalPipeline | str",
        retrieval_pipeline_2: "BaseRetrievalPipeline | str",
        rrf_k: int = 60,
        schema: Any | None = None,
        config_dir: Path | None = None,
    ):
        """Initialize Hybrid RRF retrieval pipeline.

        Args:
            session_factory: SQLAlchemy sessionmaker for database connections.
            name: Name for this pipeline.
            retrieval_pipeline_1: First retrieval pipeline (instance or name string).
            retrieval_pipeline_2: Second retrieval pipeline (instance or name string).
            rrf_k: RRF constant (default: 60). Higher values emphasize top ranks.
            schema: Schema namespace from create_schema(). If None, uses default schema.
            config_dir: Directory containing pipeline configs (for loading by name).
        """
        self.rrf_k = rrf_k
        super().__init__(
            session_factory,
            name,
            retrieval_pipeline_1,
            retrieval_pipeline_2,
            schema,
            config_dir,
        )

    def _get_pipeline_config(self) -> dict[str, Any]:
        """Return Hybrid RRF pipeline configuration."""
        return {
            "type": "hybrid_rrf",
            "retrieval_pipeline_1": self._retrieval_pipeline_1.name,
            "retrieval_pipeline_2": self._retrieval_pipeline_2.name,
            "rrf_k": self.rrf_k,
        }

    def _fuse_results(
        self,
        results_1: list[dict[str, Any]],
        results_2: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Fuse results using Reciprocal Rank Fusion."""
        return _rrf_fuse(results_1, results_2, self.rrf_k, top_k)


class HybridCCRetrievalPipeline(HybridRetrievalPipeline):
    """Hybrid retrieval pipeline using Convex Combination with score normalization.

    CC normalizes scores from both pipelines and combines them with configurable weight.

    Formula: combined = weight x norm(scores_1) + (1-weight) x norm(scores_2)

    Example:
        ```python
        from autorag_research.pipelines.retrieval import (
            HybridCCRetrievalPipeline,
            VectorSearchRetrievalPipeline,
            BM25RetrievalPipeline,
        )

        # Create with pipeline names (auto-loaded)
        hybrid = HybridCCRetrievalPipeline(
            session_factory=session_factory,
            name="hybrid_cc",
            retrieval_pipeline_1="vector_search",
            retrieval_pipeline_2="bm25",
            weight=0.6,  # 60% vector, 40% BM25
            normalize_method="mm",
        )

        results = hybrid.retrieve("What is machine learning?", top_k=10)
        ```
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        retrieval_pipeline_1: "BaseRetrievalPipeline | str",
        retrieval_pipeline_2: "BaseRetrievalPipeline | str",
        weight: float = 0.5,
        normalize_method: NormalizationMethod = "mm",
        pipeline_1_min: float | None = None,
        pipeline_2_min: float | None = None,
        schema: Any | None = None,
        config_dir: Path | None = None,
    ):
        """Initialize Hybrid CC retrieval pipeline.

        Args:
            session_factory: SQLAlchemy sessionmaker for database connections.
            name: Name for this pipeline.
            retrieval_pipeline_1: First retrieval pipeline (instance or name string).
            retrieval_pipeline_2: Second retrieval pipeline (instance or name string).
            weight: Weight for pipeline_1 scores (0-1). Default 0.5.
            normalize_method: Score normalization method ("mm", "tmm", "z", "dbsf").
            pipeline_1_min: Theoretical min score for TMM (pipeline_1).
            pipeline_2_min: Theoretical min score for TMM (pipeline_2).
            schema: Schema namespace from create_schema(). If None, uses default schema.
            config_dir: Directory containing pipeline configs (for loading by name).
        """
        self.weight = weight
        self.normalize_method = normalize_method
        self.pipeline_1_min = pipeline_1_min
        self.pipeline_2_min = pipeline_2_min
        super().__init__(
            session_factory,
            name,
            retrieval_pipeline_1,
            retrieval_pipeline_2,
            schema,
            config_dir,
        )

    def _get_pipeline_config(self) -> dict[str, Any]:
        """Return Hybrid CC pipeline configuration."""
        config = {
            "type": "hybrid_cc",
            "retrieval_pipeline_1": self._retrieval_pipeline_1.name,
            "retrieval_pipeline_2": self._retrieval_pipeline_2.name,
            "weight": self.weight,
            "normalize_method": self.normalize_method,
        }
        if self.normalize_method == "tmm":
            config["pipeline_1_min"] = self.pipeline_1_min
            config["pipeline_2_min"] = self.pipeline_2_min
        return config

    def _fuse_results(
        self,
        results_1: list[dict[str, Any]],
        results_2: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Fuse results using Convex Combination with score normalization."""
        return _cc_fuse(
            results_1,
            results_2,
            self.weight,
            top_k,
            self.normalize_method,
            self.pipeline_1_min,
            self.pipeline_2_min,
        )
