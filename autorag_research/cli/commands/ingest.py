"""ingest command - Ingest datasets into PostgreSQL using Typer CLI."""

import logging
import sys
import tempfile
from pathlib import Path
from typing import Annotated

import typer
from huggingface_hub import hf_hub_download

from autorag_research.cli.configs.db import DatabaseConfig
from autorag_research.cli.configs.ingestors import (
    INGESTOR_REGISTRY,
    IngestorSpec,
    generate_db_name,
    get_ingestor_help,
)
from autorag_research.cli.utils import load_db_config_from_yaml

logger = logging.getLogger(__name__)

HF_REPO_ID = "NomaDamas/autorag-research-datasets"

# Create ingest sub-app
ingest_app = typer.Typer(
    name="ingest",
    help="Ingest datasets into PostgreSQL.",
    no_args_is_help=True,
)


def create_ingest_command(ingestor_name: str, spec: IngestorSpec):  # noqa: C901
    """Create a Typer command function for an ingestor dynamically."""

    def ingest_command(  # noqa: C901
        dataset_value: Annotated[
            str | None,
            typer.Option(f"--{spec.cli_option}", help="Dataset to ingest. Use --list for available values."),
        ] = None,
        show_list: Annotated[bool, typer.Option("--list", help="Show all available dataset values and exit")] = False,
        # Common options
        subset: Annotated[str, typer.Option("--subset", help="Dataset split: train, dev, or test")] = "test",
        query_limit: Annotated[
            int | None, typer.Option("--query-limit", help="Maximum number of queries to ingest")
        ] = None,
        min_corpus_cnt: Annotated[
            int | None, typer.Option("--min-corpus-cnt", help="Minimum number of corpus documents to ingest")
        ] = None,
        db_name: Annotated[
            str | None, typer.Option("--db-name", help="Custom database schema name (auto-generated if not specified)")
        ] = None,
        # Database connection options
        db_host: Annotated[
            str | None, typer.Option("--db-host", help="Database host (default: from configs/db/default.yaml)")
        ] = None,
        db_port: Annotated[
            int | None, typer.Option("--db-port", help="Database port (default: from configs/db/default.yaml)")
        ] = None,
        db_user: Annotated[
            str | None, typer.Option("--db-user", help="Database user (default: from configs/db/default.yaml)")
        ] = None,
        db_password: Annotated[
            str | None, typer.Option("--db-password", help="Database password (or set PGPASSWORD)")
        ] = None,
        db_database: Annotated[
            str | None, typer.Option("--db-database", help="Database name (default: from configs/db/default.yaml)")
        ] = None,
        # Extra options for specific ingestors (handled via context)
        batch_size: Annotated[
            int | None, typer.Option("--batch-size", help="Batch size for streaming ingestion")
        ] = None,
        score_threshold: Annotated[
            int | None, typer.Option("--score-threshold", help="Minimum relevance score threshold (0-2)")
        ] = None,
        include_instruction: Annotated[
            bool | None,
            typer.Option(
                "--include-instruction/--no-include-instruction",
                help="Include instruction prefix for InstructionRetrieval tasks",
            ),
        ] = None,
        document_mode: Annotated[
            str | None, typer.Option("--document-mode", help="Document mode: 'short' or 'long'")
        ] = None,
    ) -> None:
        # Handle --list flag
        if show_list:
            typer.echo(f"\nAvailable values for --{spec.cli_option}:")
            for val in spec.available_values:
                typer.echo(f"  {val}")
            return

        # Validate dataset value
        if not dataset_value:
            typer.echo(f"Error: --{spec.cli_option} is required", err=True)
            typer.echo("Use --list to see available values", err=True)
            raise typer.Exit(1)

        # Handle list parameters (comma-separated) for ragbench and bright
        processed_value: str | list[str] = dataset_value
        if ingestor_name in ("ragbench", "bright") and dataset_value:
            processed_value = [v.strip() for v in str(dataset_value).split(",")]

        # Generate db_name
        params_for_name = {spec.dataset_param: processed_value}
        final_db_name = db_name or generate_db_name(ingestor_name, params_for_name, subset)

        # Load DB config from YAML first, then override with CLI options
        db_config = load_db_config_from_yaml()

        # Override with CLI options if provided
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

        typer.echo(f"\nIngesting dataset: {ingestor_name}")
        typer.echo(f"  {spec.cli_option}: {processed_value}")
        typer.echo(f"  subset: {subset}")
        typer.echo(f"  target schema: {final_db_name}")
        typer.echo(f"  database: {db_config.host}:{db_config.port}/{db_config.database}")
        typer.echo("=" * 60)

        # Try to download pre-built dump first
        dump_file = download_dump(final_db_name)

        if dump_file:
            restore_from_dump(db_config, dump_file, final_db_name)
        else:
            typer.echo("\nNo pre-built dump available.")
            typer.echo("Consider using pre-built dumps from HuggingFace Hub:")
            typer.echo(f"  https://huggingface.co/datasets/{HF_REPO_ID}")

    # Set docstring dynamically
    ingest_command.__doc__ = f"""{spec.description}

    Available values for --{spec.cli_option}: {", ".join(spec.available_values[:5])}...
    Use --list to see all available values.

    Examples:
      autorag-research ingest {ingestor_name} --{spec.cli_option}={spec.available_values[0] if spec.available_values else "value"}
      autorag-research ingest {ingestor_name} --list
    """

    return ingest_command


# Register all ingestors as subcommands
for name, spec in INGESTOR_REGISTRY.items():
    command_func = create_ingest_command(name, spec)
    ingest_app.command(name=name, help=spec.description)(command_func)


@ingest_app.callback(invoke_without_command=True)
def ingest_callback(ctx: typer.Context) -> None:
    """Ingest datasets into PostgreSQL.

    Choose an ingestor and specify the dataset to ingest.

    Examples:
      autorag-research ingest beir --dataset=scifact
      autorag-research ingest mrtydi --language=english --query-limit=100
      autorag-research ingest ragbench --configs=covidqa,msmarco
      autorag-research ingest beir --dataset=scifact --db-name=my_custom_schema
    """
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        typer.echo("\n" + get_ingestor_help())


def download_dump(schema_name: str) -> Path | None:
    """Download pre-built pg_dump from HuggingFace Hub."""
    dump_filename = f"{schema_name}.dump"

    typer.echo(f"\nLooking for pre-built dump: {dump_filename}")

    try:
        dump_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=dump_filename,
            repo_type="dataset",
            cache_dir=tempfile.gettempdir(),
        )
        typer.echo(f"  Downloaded: {dump_path}")
        return Path(dump_path)
    except Exception as e:
        logger.debug(f"Could not download pre-built dump: {e}")
        typer.echo("  Not found in HuggingFace Hub")
        return None


def restore_from_dump(db_config: DatabaseConfig, dump_file: Path, schema_name: str) -> None:
    """Restore database from pg_dump file."""
    from autorag_research.data.restore import restore_database

    typer.echo(f"\nRestoring from dump file: {dump_file}")

    try:
        restore_database(
            dump_file=str(dump_file),
            host=db_config.host,
            user=db_config.user,
            password=db_config.password,
            database=db_config.database,
            port=db_config.port,
            clean=False,
            create=True,
            no_owner=True,
            install_extensions=True,
        )
        typer.echo(f"\nSuccess! Schema '{schema_name}' restored.")
        typer.echo("\nNext steps:")
        typer.echo(f"  autorag-research info --schema={schema_name}")
        typer.echo(f"  autorag-research run --db-name={schema_name}")
    except Exception as e:
        logger.exception("Failed to restore database")
        typer.echo(f"\nError restoring database: {e}", err=True)
        typer.echo("\nTroubleshooting:", err=True)
        typer.echo("  1. Ensure PostgreSQL is running", err=True)
        typer.echo("  2. Check database credentials", err=True)
        typer.echo("  3. Ensure pg_restore is installed", err=True)
        sys.exit(1)
