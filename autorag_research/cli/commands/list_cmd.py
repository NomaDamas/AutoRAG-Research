"""list command - List available resources using Click CLI."""

import logging
import sys

import click

from autorag_research.cli.commands.ingest import load_db_config_from_yaml
from autorag_research.cli.configs.datasets import AVAILABLE_DATASETS
from autorag_research.cli.utils import discover_metrics, discover_pipelines

logger = logging.getLogger("AutoRAG-Research")

RESOURCE_TYPES = ["datasets", "pipelines", "metrics", "databases"]


@click.command()
@click.argument("resource", type=click.Choice(RESOURCE_TYPES))
@click.option("--db-host", default=None, help="Database host (default: from configs/db/default.yaml)")
@click.option("--db-port", type=int, default=None, help="Database port (default: from configs/db/default.yaml)")
@click.option("--db-user", default=None, help="Database user (default: from configs/db/default.yaml)")
@click.option("--db-password", default=None, help="Database password (default: from configs/db/default.yaml)")
@click.option("--db-database", default=None, help="Database name (default: from configs/db/default.yaml)")
def list_resources(
    resource: str,
    db_host: str | None,
    db_port: int | None,
    db_user: str | None,
    db_password: str | None,
    db_database: str | None,
) -> None:
    """List available resources.

    \b
    RESOURCE types:
      datasets   - Available datasets for ingestion
      pipelines  - Available pipeline configurations
      metrics    - Available evaluation metrics
      databases  - Database schemas (uses configs/db/default.yaml)

    \b
    Examples:
      autorag-research list datasets
      autorag-research list pipelines
      autorag-research list metrics
      autorag-research list databases
    """
    if resource == "datasets":
        print_datasets()
    elif resource == "pipelines":
        print_pipelines()
    elif resource == "metrics":
        print_metrics()
    elif resource == "databases":
        # Load from yaml, then override with CLI options if provided
        db_config = load_db_config_from_yaml()
        if db_host is not None:
            db_config.host = db_host
        if db_port is not None:
            db_config.port = db_port
        if db_user is not None:
            db_config.user = db_user
        if db_password is not None:
            db_config.password = db_password
        if db_database is not None:
            db_config.database = db_database
        print_databases(db_config.host, db_config.port, db_config.user, db_config.password, db_config.database)


def print_datasets() -> None:
    """Print available datasets."""
    click.echo("\nAvailable Datasets:")
    click.echo("-" * 60)
    for name, description in sorted(AVAILABLE_DATASETS.items()):
        click.echo(f"  {name:<25} {description}")
    click.echo("\nUsage: autorag-research ingest <ingestor> --dataset=<name>")


def print_pipelines() -> None:
    """Print available pipelines by scanning configs/pipelines/."""
    pipelines = discover_pipelines()
    click.echo("\nAvailable Pipelines:")
    click.echo("-" * 60)
    if pipelines:
        for name, description in sorted(pipelines.items()):
            click.echo(f"  {name:<20} {description}")
    else:
        click.echo("  No pipelines found. Run 'autorag-research init-config' first.")
    click.echo("\nSee configs/pipelines/ for configuration options")


def print_metrics() -> None:
    """Print available metrics by scanning configs/metrics/."""
    metrics = discover_metrics()
    click.echo("\nAvailable Metrics:")
    click.echo("-" * 60)
    if metrics:
        for name, description in sorted(metrics.items()):
            click.echo(f"  {name:<15} {description}")
    else:
        click.echo("  No metrics found. Run 'autorag-research init-config' first.")
    click.echo("\nSee configs/metrics/ for configuration options")


def print_databases(host: str, port: int, user: str, password: str, database: str) -> None:
    """Print database schemas."""
    from autorag_research.cli.utils import list_schemas_with_connection

    click.echo("\nDatabase Schemas:")
    click.echo("-" * 60)
    try:
        schemas = list_schemas_with_connection(host, port, user, password, database)
        if schemas:
            for schema in schemas:
                click.echo(f"  {schema}")
        else:
            click.echo("  No user schemas found.")
        click.echo(f"\nDatabase: {host}:{port}/{database}")
    except Exception:
        logger.exception("Error connecting to database. Make sure PostgreSQL is running and credentials are correct.")
        sys.exit(1)


if __name__ == "__main__":
    list_resources()
