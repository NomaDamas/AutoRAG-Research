"""Experiment-scoped leaderboard configuration."""

from dataclasses import dataclass
from typing import Literal

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from autorag_research.cli.config_resolver import ConfigResolver
from autorag_research.cli.utils import get_config_dir

MetricType = Literal["retrieval", "generation"]


@dataclass(frozen=True)
class LeaderboardScope:
    """Experiment-defined database, pipeline, and metric allowlists."""

    db_name: str
    pipeline_names: dict[str, tuple[str, ...]]
    metric_names: dict[str, tuple[str, ...]]


def scope_filters(
    scope: LeaderboardScope | None,
    metric_type: MetricType,
) -> tuple[tuple[str, ...] | None, tuple[str, ...] | None]:
    """Return pipeline and metric allowlists for one metric type."""
    if scope is None:
        return None, None
    return scope.pipeline_names.get(metric_type, ()), scope.metric_names.get(metric_type, ())


def scope_metric_types(scope: LeaderboardScope | None) -> list[MetricType]:
    """Return metric types that have both pipelines and metrics configured."""
    return [
        metric_type
        for metric_type in ("retrieval", "generation")
        if scope is None or (scope.pipeline_names.get(metric_type) and scope.metric_names.get(metric_type))
    ]


def initial_scope_metric_type(scope: LeaderboardScope | None) -> MetricType:
    """Return the initial metric type for a scoped leaderboard."""
    metric_types = scope_metric_types(scope)
    return metric_types[0] if metric_types else "retrieval"


def _config_references(config: DictConfig, section: str, metric_type: MetricType) -> list[str]:
    """Return normalized config references for one experiment section and type."""
    section_config = config.get(section, {})
    references = section_config.get(metric_type, []) if isinstance(section_config, DictConfig) else []
    if isinstance(references, str):
        return [references]
    return [str(reference) for reference in references]


def load_leaderboard_scope(config_name: str) -> LeaderboardScope:
    """Load leaderboard allowlists from an experiment YAML config."""
    config_dir = get_config_dir()
    experiment_path = config_dir / f"{config_name}.yaml"
    if not experiment_path.exists():
        raise FileNotFoundError(f"Config file not found: {experiment_path}")  # noqa: TRY003

    experiment_config = OmegaConf.load(experiment_path)
    if not isinstance(experiment_config, DictConfig):
        raise TypeError(f"Experiment config must be a YAML mapping: {experiment_path}")  # noqa: TRY003

    db_name = experiment_config.get("db_name")
    if not db_name:
        raise ValueError(f"Experiment config must define db_name: {experiment_path}")  # noqa: TRY003

    resolver = ConfigResolver(config_dir=config_dir)
    pipeline_names: dict[str, tuple[str, ...]] = {}
    metric_names: dict[str, tuple[str, ...]] = {}
    for metric_type in ("retrieval", "generation"):
        pipeline_references = _config_references(experiment_config, "pipelines", metric_type)
        resolved_pipeline_names = []
        for reference in pipeline_references:
            pipeline_config = resolver.resolve_config(["pipelines", metric_type], reference)
            resolved_pipeline_names.append(str(pipeline_config.get("name", reference)))
        pipeline_names[metric_type] = tuple(resolved_pipeline_names)

        metric_references = _config_references(experiment_config, "metrics", metric_type)
        resolved_metric_names = []
        for reference in metric_references:
            metric_config = resolver.resolve_config(["metrics", metric_type], reference)
            resolved_metric_names.append(str(instantiate(metric_config).get_metric_name()))
        metric_names[metric_type] = tuple(resolved_metric_names)

    return LeaderboardScope(
        db_name=str(db_name),
        pipeline_names=pipeline_names,
        metric_names=metric_names,
    )
