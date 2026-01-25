"""list command - List available resources using Typer CLI."""

import logging
import sys
from typing import Annotated, Literal

import typer

from autorag_research.cli.utils import discover_metrics, discover_pipelines
from autorag_research.data.registry import discover_ingestors
from autorag_research.orm.connection import DBConnection

logger = logging.getLogger("AutoRAG-Research")

ResourceType = Literal["ingestors", "pipelines", "metrics", "databases"]


RESOURCE_HANDLERS = {
    "ingestors": lambda **_: print_ingestors(),
    "pipelines": lambda **_: print_pipelines(),
    "metrics": lambda **_: print_metrics(),
    "databases": lambda **_: print_databases(),
}


def list_resources(
    resource: Annotated[
        ResourceType,
        typer.Argument(help="Resource type to list: datasets, ingestors, pipelines, metrics, or databases"),
    ],
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
    RESOURCE_HANDLERS[resource]()


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


def print_databases() -> None:
    """Print available databases on the PostgreSQL server."""
    db_conn = DBConnection.from_config()

    typer.echo("\nAvailable Databases:")
    typer.echo("-" * 60)
    try:
        databases = db_conn.get_database_names()
        if databases:
            for db in databases:
                typer.echo(f"  {db}")
        else:
            typer.echo("  No databases found.")
        typer.echo(f"\nServer: {db_conn.host}:{db_conn.port}")
    except Exception:
        logger.exception("Error connecting to database. Make sure PostgreSQL is running and credentials are correct.")
        sys.exit(1)
