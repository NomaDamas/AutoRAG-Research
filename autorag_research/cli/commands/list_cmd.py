"""list command - List available resources using Typer CLI."""

import logging
import sys
from typing import Annotated, Literal

import typer

from autorag_research.cli.utils import discover_metrics, discover_pipelines, load_db_config_from_yaml
from autorag_research.data.registry import discover_ingestors

logger = logging.getLogger("AutoRAG-Research")

ResourceType = Literal["ingestors", "pipelines", "metrics", "databases"]


RESOURCE_HANDLERS = {
    "ingestors": lambda **_: print_ingestors(),
    "pipelines": lambda **_: print_pipelines(),
    "metrics": lambda **_: print_metrics(),
}


def list_resources(
    resource: Annotated[
        ResourceType,
        typer.Argument(help="Resource type to list: datasets, ingestors, pipelines, metrics, or databases"),
    ],
    db_host: Annotated[
        str | None, typer.Option("--db-host", help="Database host (default: from configs/db.yaml)")
    ] = None,
    db_port: Annotated[
        int | None, typer.Option("--db-port", help="Database port (default: from configs/db.yaml)")
    ] = None,
    db_user: Annotated[
        str | None, typer.Option("--db-user", help="Database user (default: from configs/db.yaml)")
    ] = None,
    db_password: Annotated[
        str | None, typer.Option("--db-password", help="Database password (default: from configs/db.yaml)")
    ] = None,
) -> None:
    """List available resources.

    RESOURCE types:
      ingestors  - Available data ingestors with their parameters
      pipelines  - Available pipeline configurations
      metrics    - Available evaluation metrics

    Examples:
      autorag-research list ingestors
      autorag-research list pipelines
      autorag-research list metrics
      autorag-research list databases
    """
    if resource in RESOURCE_HANDLERS:
        RESOURCE_HANDLERS[resource]()
    elif resource == "databases":
        _print_databases_with_config(db_host, db_port, db_user, db_password)


def _print_databases_with_config(
    db_host: str | None,
    db_port: int | None,
    db_user: str | None,
    db_password: str | None,
) -> None:
    """Load DB config with CLI overrides and print databases."""
    db_config = load_db_config_from_yaml(db_host, db_port, db_user, db_password)
    print_databases(db_config.host, db_config.port, db_config.user, db_config.password)


def print_ingestors() -> None:
    """Print available ingestors and their parameters."""
    ingestors = discover_ingestors()
    typer.echo("\nAvailable Ingestors:")
    typer.echo("-" * 60)
    for name, meta in sorted(ingestors.items()):
        typer.echo(f"\n  {name}: {meta.description}")
        if meta.params:
            for param in meta.params:
                default_str = f" (default: {param.default})" if not param.required else " (required)"
                choices_str = ""
                if param.choices:
                    choices_preview = param.choices[:3]
                    if len(param.choices) > 3:
                        choices_str = f" [{', '.join(choices_preview)}...]"
                    else:
                        choices_str = f" [{', '.join(param.choices)}]"
                typer.echo(f"    --{param.cli_option}{default_str}{choices_str}")
    typer.echo("\nUsage: autorag-research ingest <ingestor> --<option>=<value>")


def print_pipelines() -> None:
    """Print available pipelines by scanning configs/pipelines/."""
    pipelines = discover_pipelines()
    typer.echo("\nAvailable Pipelines:")
    typer.echo("-" * 60)
    if pipelines:
        for subdir, configs in sorted(pipelines.items()):
            typer.echo(f"\n  [{subdir}]")
            for name, description in sorted(configs.items()):
                typer.echo(f"    {name:<18} {description}")
    else:
        typer.echo("  No pipelines found. Run 'autorag-research init-config' first.")
    typer.echo("\nUsage in experiment.yaml:")
    typer.echo("  pipelines:")
    typer.echo("    retrieval: [bm25]")
    typer.echo("    generation: [basic_rag]")


def print_metrics() -> None:
    """Print available metrics by scanning configs/metrics/."""
    metrics = discover_metrics()
    typer.echo("\nAvailable Metrics:")
    typer.echo("-" * 60)
    if metrics:
        for subdir, configs in sorted(metrics.items()):
            typer.echo(f"\n  [{subdir}]")
            for name, description in sorted(configs.items()):
                typer.echo(f"    {name:<13} {description}")
    else:
        typer.echo("  No metrics found. Run 'autorag-research init-config' first.")
    typer.echo("\nUsage in experiment.yaml:")
    typer.echo("  metrics:")
    typer.echo("    retrieval: [recall, ndcg]")
    typer.echo("    generation: [rouge]")


def print_databases(host: str, port: int, user: str, password: str) -> None:
    """Print available databases on the PostgreSQL server."""
    from autorag_research.cli.utils import list_databases_with_connection

    typer.echo("\nAvailable Databases:")
    typer.echo("-" * 60)
    try:
        databases = list_databases_with_connection(host, port, user, password)
        if databases:
            for db in databases:
                typer.echo(f"  {db}")
        else:
            typer.echo("  No databases found.")
        typer.echo(f"\nServer: {host}:{port}")
    except Exception:
        logger.exception("Error connecting to database. Make sure PostgreSQL is running and credentials are correct.")
        sys.exit(1)
