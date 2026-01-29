"""data command - Manage PostgreSQL dump files via HuggingFace Hub.

This module provides CLI commands for listing, downloading, uploading,
and restoring PostgreSQL dump files stored in HuggingFace Hub repositories.

Examples:
    autorag-research data list beir
    autorag-research data download beir scifact_openai-small
    autorag-research data restore beir scifact_openai-small
    autorag-research data dump --db-name=my_database
    autorag-research data upload ./backup.dump beir scifact_openai-small
"""

import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

console = Console()

logger = logging.getLogger("AutoRAG-Research")

data_app = typer.Typer(
    name="data",
    help="Manage PostgreSQL dump files via HuggingFace Hub.",
)


def _get_available_ingestors() -> list[str]:
    """Get list of ingestors with HuggingFace repos configured."""
    from autorag_research.data.registry import discover_ingestors

    registry = discover_ingestors()
    return sorted(name for name, meta in registry.items() if meta.hf_repo is not None)


@data_app.command(name="list")
def list_dumps_cmd(
    ingestor: Annotated[str, typer.Argument(help="Ingestor name (e.g., beir, mteb, ragbench)")],
) -> None:
    """List available dump files for an ingestor.

    Examples:
        autorag-research data list beir
        autorag-research data list mteb
    """
    from huggingface_hub.utils import RepositoryNotFoundError

    from autorag_research.data.hf_storage import list_available_dumps

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

    typer.echo(f"Available dumps for '{ingestor}':")
    for dump in sorted(dumps):
        typer.echo(f"  {dump}")


@data_app.command(name="download")
def download_dump_cmd(
    ingestor: Annotated[str, typer.Argument(help="Ingestor name (e.g., beir, mteb)")],
    filename: Annotated[str, typer.Argument(help="Dump filename without .dump extension")],
) -> None:
    """Download a dump file from HuggingFace Hub.

    Examples:
        autorag-research data download beir scifact_openai-small
        autorag-research data download mteb nfcorpus_bge-small
    """
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    from autorag_research.data.hf_storage import download_dump

    try:
        with console.status(f"[bold blue]Downloading '{filename}' from '{ingestor}'..."):
            path = download_dump(ingestor, filename)
    except KeyError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from None
    except RepositoryNotFoundError:
        typer.echo(f"Repository not found for ingestor '{ingestor}'", err=True)
        raise typer.Exit(1) from None
    except EntryNotFoundError:
        typer.echo(f"File '{filename}.dump' not found in repository", err=True)
        typer.echo(f"Use 'autorag-research data list {ingestor}' to see available dumps", err=True)
        raise typer.Exit(1) from None

    console.print(f"[green]✓[/green] Downloaded: {path}")


@data_app.command(name="restore")
def restore_dump_cmd(
    ingestor: Annotated[str, typer.Argument(help="Ingestor name (e.g., beir, mteb)")],
    filename: Annotated[str, typer.Argument(help="Dump filename without .dump extension")],
    db_name: Annotated[
        str | None,
        typer.Option("--db-name", help="Target database name (defaults to filename)"),
    ] = None,
    clean: Annotated[
        bool,
        typer.Option("--clean", help="Drop database objects before recreating"),
    ] = False,
    no_owner: Annotated[
        bool,
        typer.Option("--no-owner/--with-owner", help="Skip restoration of object ownership"),
    ] = True,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation prompts"),
    ] = False,
) -> None:
    """Download and restore a dump to PostgreSQL.

    Downloads the dump file from HuggingFace Hub (if not cached) and restores
    it to a PostgreSQL database. The database will be created if it doesn't exist.

    Examples:
        autorag-research data restore beir scifact_openai-small
        autorag-research data restore beir scifact_openai-small --db-name=my_custom_db
        autorag-research data restore mteb nfcorpus_bge-small --clean
        autorag-research data restore beir scifact_openai-small --clean --yes
    """
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    from autorag_research.data.hf_storage import download_dump
    from autorag_research.orm.connection import DBConnection

    target_db = db_name or filename

    # Confirm destructive operation
    if clean and not yes:
        confirm = typer.confirm(f"This will DROP and recreate database '{target_db}'. Continue?")
        if not confirm:
            typer.echo("Aborted.")
            raise typer.Exit(0)

    # Download dump file
    try:
        with console.status(f"[bold blue]Downloading '{filename}' from '{ingestor}'..."):
            dump_path = download_dump(ingestor, filename)
    except KeyError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from None
    except RepositoryNotFoundError:
        typer.echo(f"Repository not found for ingestor '{ingestor}'", err=True)
        raise typer.Exit(1) from None
    except EntryNotFoundError:
        typer.echo(f"File '{filename}.dump' not found in repository", err=True)
        typer.echo(f"Use 'autorag-research data list {ingestor}' to see available dumps", err=True)
        raise typer.Exit(1) from None

    console.print(f"[green]✓[/green] Downloaded: {dump_path}")

    # Restore to database
    try:
        with console.status(f"[bold blue]Restoring to database '{target_db}'..."):
            db_conn = DBConnection.from_config()
            db_conn.database = target_db
            db_conn.restore_database(dump_path, clean=clean, no_owner=no_owner)
    except FileNotFoundError:
        typer.echo("Config file not found. Run 'autorag-research init-config' first.", err=True)
        raise typer.Exit(1) from None
    except RuntimeError as e:
        typer.echo(f"Restore failed: {e}", err=True)
        raise typer.Exit(1) from None

    console.print(f"[green]✓[/green] Database '{target_db}' restored successfully")
    typer.echo("\nNext steps:")
    typer.echo(f"  autorag-research run --db-name={target_db}")


@data_app.command(name="dump")
def dump_database_cmd(
    db_name: Annotated[str, typer.Option("--db-name", help="Database name to dump")],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output file path (defaults to <db_name>.dump)"),
    ] = None,
    no_owner: Annotated[
        bool,
        typer.Option("--no-owner/--with-owner", help="Skip output of ownership commands"),
    ] = True,
) -> None:
    """Export a database to a dump file.

    Creates a PostgreSQL dump file using pg_dump in custom format,
    which can be restored with 'autorag-research data restore' or pg_restore.

    Examples:
        autorag-research data dump --db-name=beir_scifact_test
        autorag-research data dump --db-name=beir_scifact_test --output=./backup.dump
    """
    from autorag_research.orm.connection import DBConnection

    output_path = output or Path(f"{db_name}.dump")

    try:
        with console.status(f"[bold blue]Dumping database '{db_name}' to '{output_path}'..."):
            db_conn = DBConnection.from_config()
            db_conn.database = db_name
            result = db_conn.dump_database(output_path, no_owner=no_owner)
    except FileNotFoundError:
        typer.echo("Config file not found. Run 'autorag-research init-config' first.", err=True)
        raise typer.Exit(1) from None
    except RuntimeError as e:
        typer.echo(f"Dump failed: {e}", err=True)
        raise typer.Exit(1) from None

    console.print(f"[green]✓[/green] Dumped to: {result}")


@data_app.command(name="upload")
def upload_dump_cmd(
    file: Annotated[Path, typer.Argument(help="Path to the dump file to upload")],
    ingestor: Annotated[str, typer.Argument(help="Ingestor name (e.g., beir, mteb)")],
    filename: Annotated[str, typer.Argument(help="Target filename without .dump extension")],
    message: Annotated[
        str | None,
        typer.Option("--message", "-m", help="Commit message for the upload"),
    ] = None,
) -> None:
    """Upload a dump file to HuggingFace Hub.

    Requires authentication via HF_TOKEN environment variable or 'huggingface-cli login'.

    Examples:
        autorag-research data upload ./scifact.dump beir scifact_openai-small
        autorag-research data upload ./scifact.dump beir scifact_openai-small -m "Add new dump"
    """
    from huggingface_hub.utils import HfHubHTTPError

    from autorag_research.data.hf_storage import upload_dump

    if not file.exists():
        typer.echo(f"File not found: {file}", err=True)
        raise typer.Exit(1)

    try:
        with console.status(f"[bold blue]Uploading '{file}' to '{ingestor}/{filename}.dump'..."):
            url = upload_dump(file, ingestor, filename, commit_message=message)
    except KeyError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from None
    except HfHubHTTPError as e:
        if "401" in str(e) or "403" in str(e):
            typer.echo("Authentication required. Set HF_TOKEN or run 'huggingface-cli login'", err=True)
        else:
            typer.echo(f"Upload failed: {e}", err=True)
        raise typer.Exit(1) from None

    console.print(f"[green]✓[/green] Uploaded: {url}")
