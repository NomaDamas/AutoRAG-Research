"""Typer-based CLI application for AutoRAG-Research."""

import logging
from importlib.metadata import version as get_version
from pathlib import Path
from typing import Annotated

import typer

import autorag_research.cli as cli
from autorag_research.cli.commands.data import data_app
from autorag_research.cli.commands.ingest import ingest_app
from autorag_research.cli.commands.init import init
from autorag_research.cli.commands.plugin import plugin_app
from autorag_research.cli.commands.run import run_command
from autorag_research.cli.commands.show import show_resources

# Configure logging for CLI output
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        typer.echo(f"autorag-research {get_version('autorag-research')}")
        raise typer.Exit()


# Main Typer app
app = typer.Typer(
    name="autorag-research",
    help="AutoRAG-Research CLI - RAG research on steroids.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def main_callback(
    config_path: Annotated[
        Path | None,
        typer.Option(
            "--config-path",
            "-cp",
            help="Path to configuration directory",
            envvar="AUTORAG_RESEARCH_CONFIG_PATH",
        ),
    ] = None,
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-V",
            help="Show version and exit",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """AutoRAG-Research CLI - RAG research on steroids.

    Global options are processed before any command.
    """
    # Set global config path (default: ./configs)
    cli.CONFIG_PATH = (config_path or Path.cwd() / "configs").resolve()


# Add sub-apps
app.add_typer(data_app, name="data")
app.add_typer(ingest_app, name="ingest")
app.add_typer(plugin_app, name="plugin")

# Add simple commands
app.command(name="show")(show_resources)
app.command(name="init")(init)
app.command(name="run")(run_command)


def main() -> None:
    """CLI entry point."""
    app()


if __name__ == "__main__":
    main()
