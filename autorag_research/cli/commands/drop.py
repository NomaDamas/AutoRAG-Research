"""drop command - Drop PostgreSQL databases."""

from typing import Annotated

import typer

SYSTEM_DATABASES = {"postgres", "template0", "template1"}

drop_app = typer.Typer(
    name="drop",
    help="Drop PostgreSQL databases.",
)


@drop_app.command(name="database")
def drop_database_cmd(
    db_name: Annotated[str, typer.Option("--db-name", help="Database name to drop")],
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation prompts"),
    ] = False,
) -> None:
    """Drop a PostgreSQL database.

    Examples:
        autorag-research drop database --db-name=beir_scifact_test
        autorag-research drop database --db-name=beir_scifact_test --yes
    """
    from autorag_research.orm.connection import DBConnection

    if db_name in SYSTEM_DATABASES:
        typer.echo(f"Refusing to drop protected system database '{db_name}'.", err=True)
        raise typer.Exit(1)

    if not yes:
        confirm = typer.confirm(f"This will permanently DROP database '{db_name}'. Continue?")
        if not confirm:
            typer.echo("Aborted.")
            raise typer.Exit(0)

    try:
        db_conn = DBConnection.from_config()
        db_conn.database = db_name
        db_conn.terminate_connections()
        db_conn.drop_database()
    except FileNotFoundError:
        typer.echo("Config file not found. Run 'autorag-research init' first.", err=True)
        raise typer.Exit(1) from None
    except (RuntimeError, ValueError) as e:
        typer.echo(f"Drop failed: {e}", err=True)
        raise typer.Exit(1) from None

    typer.echo(f"Database '{db_name}' dropped successfully.")
