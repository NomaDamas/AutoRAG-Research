"""Typer-based CLI application for AutoRAG-Research."""

import logging
from pathlib import Path
from typing import Annotated

import typer

import autorag_research.cli as cli
from autorag_research.cli.commands.ingest import ingest_app
from autorag_research.cli.commands.init_config import init_config
from autorag_research.cli.commands.list_cmd import list_resources
from autorag_research.cli.commands.run import run_command
from autorag_research.cli.configs import register_configs

# Configure logging for CLI output
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)

# Main Typer app
app = typer.Typer(
    name="autorag-research",
    help="AutoRAG-Research CLI - Automate your RAG research workflows",
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
) -> None:
    """AutoRAG-Research CLI - Automate your RAG research workflows.

    Global options are processed before any command.
    """
    # Set global config path (default: ./configs)
    cli.CONFIG_PATH = (config_path or Path.cwd() / "configs").resolve()
    register_configs()


# Add ingest as a sub-app
app.add_typer(ingest_app, name="ingest")

# Add simple commands
app.command(name="list")(list_resources)
app.command(name="init-config")(init_config)
app.command(name="run")(run_command)


def main() -> None:
    """CLI entry point."""
    app()


if __name__ == "__main__":
    main()
