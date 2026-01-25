"""run command - Execute experiment pipelines with metrics evaluation."""

import logging
import os
from pathlib import Path
from typing import Annotated

import typer
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import autorag_research.cli as cli
from autorag_research.cli.utils import setup_logging
from autorag_research.config import ExecutorConfig
from autorag_research.executor import Executor
from autorag_research.orm.schema_factory import create_schema

logger = logging.getLogger("AutoRAG-Research")

# Hydra overrides to disable output directory and log file creation
HYDRA_OVERRIDES = [
    "hydra.run.dir=.",
    "hydra.output_subdir=null",
    "hydra.job.chdir=False",
    "hydra/job_logging=none",
    "hydra/hydra_logging=none",
]


def run_command(
    db_name: Annotated[
        str | None,
        typer.Option("--db-name", "-d", help="Database schema name (required)"),
    ] = None,
    config_name: Annotated[
        str,
        typer.Option("--config-name", "-cn", help="Config file name without .yaml extension"),
    ] = "experiment",
    max_retries: Annotated[
        int | None,
        typer.Option("--max-retries", help="Maximum retries for failed operations"),
    ] = None,
    eval_batch_size: Annotated[
        int | None,
        typer.Option("--eval-batch-size", help="Batch size for evaluation"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose logging"),
    ] = False,
    overrides: Annotated[
        list[str] | None,
        typer.Argument(help="Additional Hydra overrides (e.g., pipelines.0.k1=1.2)"),
    ] = None,
) -> None:
    """Run experiment pipelines with metrics evaluation.

    Configuration is loaded from configs/experiment.yaml (or specified --config-name).
    Pipelines and metrics are instantiated via _target_ in YAML.

    Examples:
      autorag-research run --db-name=beir_scifact_test
      autorag-research run --db-name=beir_scifact_test --max-retries=5
      autorag-research run --db-name=beir_scifact_test --config-name=my_experiment
    """
    setup_logging(verbose=verbose)

    # Build Hydra overrides from Typer options
    hydra_overrides = list(HYDRA_OVERRIDES)

    if db_name:
        hydra_overrides.append(f"db_name={db_name}")
    if max_retries is not None:
        hydra_overrides.append(f"max_retries={max_retries}")
    if eval_batch_size is not None:
        hydra_overrides.append(f"eval_batch_size={eval_batch_size}")

    # Add user-provided Hydra overrides
    if overrides:
        hydra_overrides.extend(overrides)

    # Get config path
    config_path = cli.CONFIG_PATH
    if config_path is None:
        typer.echo("Error: config path not set", err=True)
        raise typer.Exit(1)

    # Load and resolve configuration
    cfg = _load_experiment_config(
        config_path=config_path,
        config_name=config_name,
        hydra_overrides=hydra_overrides,
        db_name=db_name,
        max_retries=max_retries,
        eval_batch_size=eval_batch_size,
    )

    try:
        _run_experiment(cfg)
    except typer.Exit:
        raise
    except Exception as e:
        logger.exception("Failed to run experiment")
        typer.echo(f"\nError: {e}", err=True)
        raise typer.Exit(1) from None


def _load_experiment_config(
    config_path: Path,
    config_name: str,
    hydra_overrides: list[str],
    db_name: str | None,
    max_retries: int | None,
    eval_batch_size: int | None,
) -> DictConfig:
    """Load experiment config, detecting dict vs Hydra syntax."""
    experiment_yaml_path = config_path / f"{config_name}.yaml"
    if not experiment_yaml_path.exists():
        typer.echo(f"Error: Config file not found: {experiment_yaml_path}", err=True)
        raise typer.Exit(1)

    experiment_cfg = OmegaConf.load(experiment_yaml_path)

    # Ensure we have a DictConfig (not a list)
    if not isinstance(experiment_cfg, DictConfig):
        typer.echo("Error: experiment config must be a YAML mapping, not a list", err=True)
        raise typer.Exit(1)

    # Check if using new dict-based syntax
    if _is_dict_syntax(experiment_cfg):
        return _build_config_from_dict(
            experiment_cfg=experiment_cfg,
            config_path=config_path,
            db_name=db_name,
            max_retries=max_retries,
            eval_batch_size=eval_batch_size,
        )

    # Legacy Hydra defaults syntax
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_path), version_base=None):
        return compose(config_name=config_name, overrides=hydra_overrides)


def _is_dict_syntax(cfg: DictConfig) -> bool:
    """Check if config uses the new dict-based syntax.

    New syntax has pipelines as a dict with subdirectory keys:
        pipelines:
          retrieval: [bm25]
          generation: [basic_rag]

    Legacy syntax has pipelines as a list with _target_ in each item.
    """
    from omegaconf import ListConfig

    pipelines = cfg.get("pipelines", {})
    # New syntax: pipelines is a dict with string keys (subdirectories)
    if isinstance(pipelines, DictConfig):
        # Check if any value is a list of strings (names) or string rather than having _target_
        for value in pipelines.values():
            # OmegaConf creates ListConfig for lists, not Python list
            if isinstance(value, (list, tuple, ListConfig, str)):
                return True
            if isinstance(value, DictConfig) and not hasattr(value, "_target_"):
                return True
    return False


def _build_config_from_dict(
    experiment_cfg: DictConfig,
    config_path: Path,
    db_name: str | None,
    max_retries: int | None,
    eval_batch_size: int | None,
) -> DictConfig:
    """Build config from dict-based syntax using ConfigResolver.

    Args:
        experiment_cfg: Raw experiment YAML config.
        config_path: Path to configs directory.
        db_name: Override for db_name.
        max_retries: Override for max_retries.
        eval_batch_size: Override for eval_batch_size.

    Returns:
        Resolved DictConfig with all configs loaded.
    """
    from autorag_research.cli.config_resolver import ConfigResolver

    resolver = ConfigResolver(config_dir=config_path)

    # Load database config
    db_cfg = resolver.load_db_config("default")

    # Resolve pipeline and metric configs
    pipeline_cfgs = resolver.resolve_pipelines(experiment_cfg.get("pipelines", {}))
    metric_cfgs = resolver.resolve_metrics(experiment_cfg.get("metrics", {}))

    # Build final config with CLI overrides taking precedence
    return OmegaConf.create({
        "db": db_cfg,
        "db_name": db_name or experiment_cfg.get("db_name"),
        "max_retries": max_retries if max_retries is not None else experiment_cfg.get("max_retries", 3),
        "eval_batch_size": eval_batch_size
        if eval_batch_size is not None
        else experiment_cfg.get("eval_batch_size", 100),
        "pipelines": pipeline_cfgs,
        "metrics": metric_cfgs,
    })


def _validate_config(cfg: DictConfig) -> tuple[str, list, list]:
    """Validate configuration and return db_name, pipelines, and metrics.

    Raises:
        typer.Exit: If validation fails.
    """
    db_name = cfg.get("db_name", "")
    if not db_name:
        typer.echo("Error: db_name is required in config")
        typer.echo("Add 'db_name: <db_name>' to your experiment YAML or pass --db-name=xxx")
        raise typer.Exit(1)

    pipelines_cfg = cfg.get("pipelines", [])
    if not pipelines_cfg:
        typer.echo("Error: at least one pipeline is required")
        typer.echo("Add pipelines to your experiment YAML using defaults")
        raise typer.Exit(1)

    return db_name, pipelines_cfg, cfg.get("metrics", [])


def _print_config_summary(cfg: DictConfig, db_name: str, pipelines_cfg: list, metrics_cfg: list) -> None:
    """Print configuration summary to console."""
    typer.echo("\nAutoRAG-Research Experiment Runner")
    typer.echo("=" * 60)

    typer.echo(f"\nDatabase name: {db_name}")
    typer.echo(f"Database: {cfg.db.host}:{cfg.db.port}/{cfg.db.database}")

    typer.echo("\nPipelines:")
    for i, p in enumerate(pipelines_cfg):
        if hasattr(p, "_target_"):
            typer.echo(f"  [{i}] {p._target_.split('.')[-1]}")
        typer.echo(f"       {OmegaConf.to_yaml(p).strip()}")

    typer.echo("\nMetrics:")
    for i, m in enumerate(metrics_cfg):
        if hasattr(m, "_target_"):
            typer.echo(f"  [{i}] {m._target_.split('.')[-1]}")
        typer.echo(f"       {OmegaConf.to_yaml(m).strip()}")

    typer.echo("\nConfiguration:")
    typer.echo(f"  max_retries: {cfg.get('max_retries', 3)}")
    typer.echo(f"  eval_batch_size: {cfg.get('eval_batch_size', 100)}")


def _determine_embedding_dim(db_url: str, db_name: str) -> int:
    """Auto-detect embedding dimension from DB.

    Returns:
        Embedding dimension to use.
    """
    from autorag_research.util import detect_embedding_dimension

    detected_dim = detect_embedding_dimension(db_url, db_name)
    if detected_dim is not None:
        typer.echo(f"  embedding_dim: {detected_dim} (auto-detected from DB)")
        return detected_dim

    typer.echo("  embedding_dim: 1536 (default)")
    return 1536


def _run_experiment(cfg: DictConfig) -> None:
    """Execute the experiment with the given configuration."""
    db_name, pipelines_cfg, metrics_cfg = _validate_config(cfg)
    _print_config_summary(cfg, db_name, pipelines_cfg, metrics_cfg)

    # Build executor config using instantiate
    try:
        executor_config = build_executor_config(cfg)
    except Exception as e:
        logger.exception("Failed to build executor config")
        typer.echo(f"\nError building configuration: {e}")
        raise typer.Exit(1) from None

    # Create database session
    try:
        session_factory, db_url = create_session_factory(cfg)
    except Exception as e:
        logger.exception("Failed to connect to database")
        typer.echo(f"\nError connecting to database: {e}")
        raise typer.Exit(1) from None

    embedding_dim = _determine_embedding_dim(db_url, db_name)
    schema = create_schema(dim=embedding_dim)

    typer.echo("\n" + "-" * 60)
    typer.echo("Starting experiment...")
    typer.echo("-" * 60)

    try:
        executor = Executor(
            session_factory=session_factory,
            config=executor_config,
            schema=schema,
        )
        result = executor.run()
        print_results(result)
    except Exception as e:
        logger.exception("Experiment failed")
        typer.echo(f"\nExperiment failed: {e}")
        raise typer.Exit(1) from None


def build_executor_config(cfg: DictConfig) -> ExecutorConfig:
    """Build ExecutorConfig from Hydra configuration using instantiate."""
    pipelines = []
    metrics = []

    # Instantiate pipeline configs via _target_
    for p_cfg in cfg.get("pipelines", []):
        pipeline_config = instantiate(p_cfg)
        pipelines.append(pipeline_config)

    # Instantiate metric configs via _target_
    for m_cfg in cfg.get("metrics", []):
        metric_config = instantiate(m_cfg)
        metrics.append(metric_config)

    return ExecutorConfig(
        pipelines=pipelines,
        metrics=metrics,
        max_retries=cfg.get("max_retries", 3),
        eval_batch_size=cfg.get("eval_batch_size", 100),
    )


def create_session_factory(cfg: DictConfig) -> tuple:
    """Create SQLAlchemy session factory from config.

    Returns:
        Tuple of (session_factory, db_url) for use in session creation and dimension detection.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    password = cfg.db.password
    if isinstance(password, str) and password.startswith("${") and password.endswith("}"):
        password = os.environ.get("PGPASSWORD", "postgres")

    db_url = f"postgresql+psycopg://{cfg.db.user}:{password}@{cfg.db.host}:{cfg.db.port}/{cfg.db.database}"
    engine = create_engine(db_url)
    return sessionmaker(bind=engine), db_url


def print_results(result) -> None:
    """Print experiment results."""
    typer.echo("\n" + "=" * 60)
    typer.echo("EXPERIMENT RESULTS")
    typer.echo("=" * 60)

    typer.echo("\nPipeline Results:")
    typer.echo("-" * 60)
    for pr in result.pipeline_results:
        status = "+" if pr.success else "x"
        typer.echo(f"  {status} {pr.pipeline_name}")
        if pr.error:
            typer.echo(f"      Error: {pr.error}")

    typer.echo("\nMetric Results:")
    typer.echo("-" * 60)
    for mr in result.metric_results:
        typer.echo(f"  {mr.metric_name} @ {mr.pipeline_name}")
        typer.echo(f"      Score: {mr.score:.4f}")

    successful_pipelines = sum(1 for pr in result.pipeline_results if pr.success)
    total_pipelines = len(result.pipeline_results)
    typer.echo(f"\nSummary: {successful_pipelines}/{total_pipelines} pipelines completed")

    if result.metric_results:
        avg_score = sum(mr.score for mr in result.metric_results) / len(result.metric_results)
        typer.echo(f"Average metric score: {avg_score:.4f}")
