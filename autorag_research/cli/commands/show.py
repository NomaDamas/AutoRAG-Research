"""show command - Show available resources using Typer CLI."""

import logging
import sys
from collections.abc import Callable
from typing import Annotated, Any, Literal

import typer

from autorag_research.cli.utils import discover_metrics, discover_pipelines
from autorag_research.data.registry import discover_ingestors
from autorag_research.orm.connection import DBConnection

logger = logging.getLogger("AutoRAG-Research")

ResourceType = Literal["datasets", "ingestors", "pipelines", "metrics", "databases"]

PIPELINE_TYPES = ("retrieval", "generation")


RESOURCE_HANDLERS: dict[str, Callable[..., Any]] = {
    "datasets": lambda name=None: print_datasets(name),
    "ingestors": lambda **_: print_ingestors(),
    "pipelines": lambda **_: print_pipelines(),
    "metrics": lambda **_: print_metrics(),
    "databases": lambda **_: print_databases(),
}


def show_resources(
    resource: Annotated[
        ResourceType,
        typer.Argument(help="Resource type: datasets, ingestors, pipelines, metrics, or databases"),
    ],
    name: Annotated[
        str | None,
        typer.Argument(help="Resource name (e.g., ingestor name for 'datasets')"),
    ] = None,
) -> None:
    """Show available resources.

    RESOURCE types:
      datasets   - Available dump files (optionally filter by ingestor name)
      ingestors  - Available data ingestors with their parameters
      pipelines  - Available pipeline configurations
      metrics    - Available evaluation metrics
      databases  - Available PostgreSQL databases

    Examples:
      autorag-research show datasets
      autorag-research show datasets beir
      autorag-research show ingestors
      autorag-research show pipelines
      autorag-research show metrics
      autorag-research show databases
    """
    RESOURCE_HANDLERS[resource](name=name)


def print_datasets(ingestor: str | None = None) -> None:
    """Print available datasets (dump files).

    If ingestor is provided, shows dumps for that specific ingestor.
    Otherwise, shows all ingestors that have HuggingFace repos configured.
    """
    from huggingface_hub.utils import RepositoryNotFoundError

    from autorag_research.data.hf_storage import list_available_dumps

    if ingestor is None:
        # Show all ingestors with HF repos
        registry = discover_ingestors()
        ingestors_with_hf = sorted(name for name, meta in registry.items() if meta.hf_repo is not None)

        typer.echo("\nIngestors list : show available file names with 'autorag-research show datasets <ingestor>'")
        typer.echo("-" * 60)
        if ingestors_with_hf:
            for name in ingestors_with_hf:
                typer.echo(f"  {name}")
        else:
            typer.echo("  No ingestors with HuggingFace repos found.")
        typer.echo("\nUsage: autorag-research show datasets <ingestor>")
        return

    # Show dumps for specific ingestor
    try:
        dumps = list_available_dumps(ingestor)
    except KeyError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from None
    except RepositoryNotFoundError:
        typer.echo(f"Repository not found for ingestor '{ingestor}'", err=True)
        raise typer.Exit(1) from None

    if not dumps:
        typer.echo(f"No dump files found for '{ingestor}'")
        return

    typer.echo(f"\nAvailable dumps for '{ingestor}':")
    typer.echo("-" * 60)
    for dump in sorted(dumps):
        typer.echo(f"  {dump}")


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
    typer.echo("\nAvailable Pipelines:")
    typer.echo("-" * 60)

    has_pipelines = False
    for pipeline_type in PIPELINE_TYPES:
        pipelines = discover_pipelines(pipeline_type)
        if pipelines:
            has_pipelines = True
            typer.echo(f"\n  [{pipeline_type}]")
            for name, description in sorted(pipelines.items()):
                typer.echo(f"    {name:<18} {description}")

    if not has_pipelines:
        typer.echo("  No pipelines found. Run 'autorag-research init' first.")
    typer.echo("\nExample Usage in experiment.yaml:")
    typer.echo("  pipelines:")
    typer.echo("    retrieval: [bm25]")
    typer.echo("    generation: [basic_rag]")


def print_metrics() -> None:
    """Print available metrics by scanning configs/metrics/."""
    typer.echo("\nAvailable Metrics:")
    typer.echo("-" * 60)

    has_metrics = False
    for pipeline_type in ("retrieval", "generation"):
        metrics = discover_metrics(pipeline_type)
        if metrics:
            has_metrics = True
            typer.echo(f"\n  [{pipeline_type}]")
            for name, description in sorted(metrics.items()):
                typer.echo(f"    {name:<13} {description}")

    if not has_metrics:
        typer.echo("  No metrics found. Run 'autorag-research init' first.")
    typer.echo("\nUExample sage in experiment.yaml:")
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
