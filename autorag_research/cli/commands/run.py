"""run command - Execute experiment pipelines with metrics evaluation."""

import logging
import os
from typing import Annotated

import typer
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from autorag_research.cli.config_resolver import ConfigResolver
from autorag_research.cli.utils import get_config_dir, setup_logging
from autorag_research.config import ExecutorConfig
from autorag_research.executor import Executor
from autorag_research.orm.connection import DBConnection

# Register hydra resolver for standalone use (outside Hydra main context)
OmegaConf.register_new_resolver("hydra", lambda x: os.getcwd() if x == "runtime.cwd" else None, replace=True)

logger = logging.getLogger("AutoRAG-Research")


def run_command(  # noqa: C901
    db_name: Annotated[
        str | None,
        typer.Option("--db-name", "-d", help="Database schema name (required)"),
    ] = None,
    config_name: Annotated[
        str,
        typer.Option("--config-name", "-cn", help="Config file name without .yaml extension"),
    ] = "experiment",
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose logging"),
    ] = False,
) -> None:
    """Run experiment pipelines with metrics evaluation.

    Configuration is loaded from configs/experiment.yaml (or specified --config-name).

    Examples:
      autorag-research run --db-name=beir_scifact_test
      autorag-research run --db-name=beir_scifact_test --verbose
      autorag-research run --db-name=beir_scifact_test --config-name=my_experiment
    """
    setup_logging(verbose=verbose)

    # 1. Get config path
    config_path = get_config_dir()
    if config_path is None:
        typer.echo("Error: config path not set", err=True)
        raise typer.Exit(1)

    # 2. Load experiment YAML
    experiment_yaml_path = config_path / f"{config_name}.yaml"
    if not experiment_yaml_path.exists():
        typer.echo(f"Error: Config file not found: {experiment_yaml_path}", err=True)
        raise typer.Exit(1)

    experiment_cfg = OmegaConf.load(experiment_yaml_path)
    if not isinstance(experiment_cfg, DictConfig):
        typer.echo("Error: experiment config must be a YAML mapping, not a list", err=True)
        raise typer.Exit(1)

    # 3. Resolve db_name (CLI override takes precedence)
    resolved_db_name = db_name or experiment_cfg.get("db_name")
    if not resolved_db_name:
        typer.echo("Error: db_name is required in config")
        typer.echo("Add 'db_name: <db_name>' to your experiment YAML or pass --db-name=xxx")
        raise typer.Exit(1)

    # 4. Database connection
    db_conn = DBConnection.from_config(config_path)
    db_conn.database = resolved_db_name

    # 5. Resolve pipeline and metric configs
    resolver = ConfigResolver(config_dir=config_path)
    pipeline_cfgs = resolver.resolve_pipelines(experiment_cfg.get("pipelines", {}))
    metric_cfgs = resolver.resolve_metrics(experiment_cfg.get("metrics", {}))

    if not pipeline_cfgs:
        typer.echo("Error: at least one pipeline is required")
        typer.echo("Add pipelines to your experiment YAML using defaults")
        raise typer.Exit(1)

    typer.echo("\nAutoRAG-Research Experiment Runner")
    typer.echo("=" * 60)

    typer.echo(f"\nDatabase name: {db_name}")
    typer.echo(f"Database: {db_conn.host}:{db_conn.port}/{db_conn.database}")

    typer.echo("\nPipelines:")
    for i, p in enumerate(pipeline_cfgs):
        if hasattr(p, "_target_"):
            typer.echo(f"  [{i}] {p._target_.split('.')[-1]}")
        typer.echo(f"       {OmegaConf.to_yaml(p).strip()}")

    typer.echo("\nMetrics:")
    for i, m in enumerate(metric_cfgs):
        if hasattr(m, "_target_"):
            typer.echo(f"  [{i}] {m._target_.split('.')[-1]}")
        typer.echo(f"       {OmegaConf.to_yaml(m).strip()}")

    resolved_max_retries = experiment_cfg.get("max_retries", 3)
    resolved_eval_batch_size = experiment_cfg.get("eval_batch_size", 100)

    typer.echo("\nConfiguration:")
    typer.echo(f"  max_retries: {resolved_max_retries}")
    typer.echo(f"  eval_batch_size: {resolved_eval_batch_size}")

    # 7. Build ExecutorConfig
    executor_config = build_executor_config(
        pipeline_cfgs, metric_cfgs, max_retries=resolved_max_retries, eval_batch_size=resolved_eval_batch_size
    )

    # 8. Get schema and session factory (using new get_schema method)
    try:
        schema = db_conn.get_schema()
        session_factory = db_conn.get_session_factory()
    except Exception as e:
        logger.exception("Failed to connect to database")
        typer.echo(f"\nError connecting to database: {e}", err=True)
        raise typer.Exit(1) from None

    # 9. Run experiment
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
        typer.echo(f"\nExperiment failed: {e}", err=True)
        raise typer.Exit(1) from None


def build_executor_config(
    pipelines: list, metrics: list, max_retries: int = 3, eval_batch_size: int = 100
) -> ExecutorConfig:
    """Build ExecutorConfig from Hydra configuration using instantiate."""
    instantiated_pipelines = [instantiate(p_cfg) for p_cfg in pipelines]
    instantiated_metrics = [instantiate(m_cfg) for m_cfg in metrics]

    return ExecutorConfig(
        pipelines=instantiated_pipelines,
        metrics=instantiated_metrics,
        max_retries=max_retries,
        eval_batch_size=eval_batch_size,
    )


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
        if pr.error_message:
            typer.echo(f"      Error: {pr.error_message}")

    typer.echo("\nMetric Results:")
    typer.echo("-" * 60)
    for mr in result.metric_results:
        typer.echo(f"  {mr.metric_name} @ {mr.pipeline_id}")
        typer.echo(f"      Score: {mr.average:.4f}")

    successful_pipelines = sum(1 for pr in result.pipeline_results if pr.success)
    total_pipelines = len(result.pipeline_results)
    typer.echo(f"\nSummary: {successful_pipelines}/{total_pipelines} pipelines completed")

    if result.metric_results:
        avg_score = sum(mr.average for mr in result.metric_results) / len(result.metric_results)
        typer.echo(f"Average metric score: {avg_score:.4f}")
