"""Main entry point for AutoRAG-Research CLI."""

import logging
import sys
from pathlib import Path

from autorag_research.cli.configs import register_configs

# Configure logging for CLI output
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)

# Hydra overrides to disable output directory and log file creation
HYDRA_OVERRIDES = [
    "hydra.run.dir=.",
    "hydra.output_subdir=null",
    "hydra.job.chdir=False",
    "hydra/job_logging=none",
    "hydra/hydra_logging=none",
]


def extract_config_path_option(args: list[str]) -> tuple[list[str], Path]:
    """Extract --config-path option from args and return filtered args + config_path.

    Supports --config-path=value, --config-path value, -cp=value, and -cp value formats.

    Args:
        args: Command line arguments (without program name).

    Returns:
        Tuple of (filtered_args, config_path). config_path defaults to CWD/configs.
    """
    config_path = Path.cwd() / "configs"
    filtered_args = []
    skip_next = False

    for i, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue

        if arg.startswith("--config-path=") or arg.startswith("-cp="):
            config_path = Path(arg.split("=", 1)[1])
        elif arg in ("--config-path", "-cp"):
            if i + 1 < len(args):
                config_path = Path(args[i + 1])
                skip_next = True
        else:
            filtered_args.append(arg)

    return filtered_args, config_path


def main() -> None:
    """Main CLI entry point that dispatches to subcommands."""
    import autorag_research.cli as cli

    # Extract --config-path from all args first (before command)
    all_args = sys.argv[1:]
    all_args, config_path = extract_config_path_option(all_args)

    # Set global config path
    cli.CONFIG_PATH = config_path.resolve()

    register_configs()

    if len(all_args) < 1:
        print_usage()
        sys.exit(1)

    command = all_args[0]
    remaining_args = all_args[1:]

    # Handle Click-based commands separately
    # Click handles its own argv parsing, uses ConfigPathManager
    if command == "ingest":
        from autorag_research.cli.commands.ingest import ingest

        sys.argv = [f"{sys.argv[0]} ingest", *remaining_args]
        ingest()
        return

    if command == "list":
        from autorag_research.cli.commands.list_cmd import list_resources

        sys.argv = [f"{sys.argv[0]} list", *remaining_args]
        list_resources()
        return

    # For Hydra-based commands, inject --config-path for Hydra to use
    # and inject Hydra overrides to disable output directory
    hydra_config_path = f"--config-path={config_path.resolve()}"

    sys.argv = [sys.argv[0], hydra_config_path, *remaining_args, *HYDRA_OVERRIDES]

    if command == "init-config":
        from autorag_research.cli.commands.init_config import init_config

        init_config()
    elif command == "run":
        from autorag_research.cli.commands.run import run

        run()
    elif command in {"--help", "-h", "help"}:
        print_usage()
    else:
        print(f"Unknown command: {command}")
        print_usage()
        sys.exit(1)


def print_usage() -> None:
    """Print CLI usage information."""
    usage = """
AutoRAG-Research CLI - Automate your RAG research workflows

Usage: autorag-research [--config-path <path>] <command> [options]

Global Options:
  --config-path, -cp  Path to configuration directory (default: ./configs)

Commands:
  init-config    Download default configuration files to config directory
  ingest         Ingest a dataset into PostgreSQL
  run            Run experiment pipelines with metrics evaluation
  list           List available resources (datasets, pipelines, metrics)

Examples:
  # 1. Download configuration files (first time setup)
  autorag-research init-config

  # 2. Use custom config directory
  autorag-research --config-path=/my/configs run db_name=test

  # 3. Ingest datasets (see available ingestors with: ingest --help)
  autorag-research ingest beir --dataset=scifact
  autorag-research ingest mrtydi --language=english --query-limit=100
  autorag-research ingest ragbench --configs=covidqa,msmarco

  # 4. List available resources
  autorag-research list datasets
  autorag-research list pipelines
  autorag-research list metrics

  # 5. Run experiment (uses configs/experiment.yaml)
  autorag-research run db_name=beir_scifact_test

  # 6. Run with overrides
  autorag-research run db_name=beir_scifact_test pipelines.0.k1=1.2

For more information, visit: https://github.com/vkehfdl1/AutoRAG-Research
"""
    print(usage)


if __name__ == "__main__":
    main()
