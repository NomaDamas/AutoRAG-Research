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


def get_config_path() -> str | None:
    """Get the config path if configs/ directory exists."""
    configs_dir = Path.cwd() / "configs"
    if configs_dir.exists():
        return str(configs_dir)
    return None


def extract_db_name_option(args: list[str]) -> tuple[list[str], str | None]:
    """Extract --db-name option from args and return filtered args + db_name value.

    Supports both --db-name=value and --db-name value formats.
    """
    db_name = None
    filtered_args = []
    skip_next = False

    for i, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue

        if arg.startswith("--db-name="):
            db_name = arg.split("=", 1)[1]
        elif arg == "--db-name":
            if i + 1 < len(args):
                db_name = args[i + 1]
                skip_next = True
        else:
            filtered_args.append(arg)

    return filtered_args, db_name


def main() -> None:
    """Main CLI entry point that dispatches to subcommands."""
    register_configs()

    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    command = sys.argv[1]

    # Handle Click-based commands separately
    # Click handles its own argv parsing
    if command == "ingest":
        from autorag_research.cli.commands.ingest import ingest

        sys.argv = [f"{sys.argv[0]} ingest", *sys.argv[2:]]
        ingest()
        return

    if command == "list":
        from autorag_research.cli.commands.list_cmd import list_resources

        sys.argv = [f"{sys.argv[0]} list", *sys.argv[2:]]
        list_resources()
        return

    # For Hydra-based commands, remove the command from argv
    # and inject Hydra overrides to disable output directory
    remaining_args = sys.argv[2:]

    # Handle --db-name for run command (convert to Hydra schema override)
    if command == "run":
        remaining_args, db_name_override = extract_db_name_option(remaining_args)
        if db_name_override:
            remaining_args.append(f"schema={db_name_override}")

    sys.argv = [sys.argv[0], *remaining_args, *HYDRA_OVERRIDES]

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

Usage: autorag-research <command> [options]

Commands:
  init-config    Download default configuration files to ./configs/
  ingest         Ingest a dataset into PostgreSQL
  run            Run experiment pipelines with metrics evaluation
  list           List available resources (datasets, pipelines, metrics)

Examples:
  # 1. Download configuration files (first time setup)
  autorag-research init-config

  # 2. Ingest datasets (see available ingestors with: ingest --help)
  autorag-research ingest beir --dataset=scifact
  autorag-research ingest mrtydi --language=english --query-limit=100
  autorag-research ingest ragbench --configs=covidqa,msmarco

  # 3. List available resources
  autorag-research list datasets
  autorag-research list pipelines
  autorag-research list metrics

  # 4. Run experiment (uses configs/experiment.yaml)
  autorag-research run --db-name=beir_scifact_test

  # 5. Run with overrides
  autorag-research run --db-name=beir_scifact_test pipelines.0.k1=1.2

  # 6. Multirun (hyperparameter sweep)
  autorag-research run --db-name=beir_scifact_test -m pipelines.0.k1=0.5,0.9,1.2

For more information, visit: https://github.com/vkehfdl1/AutoRAG-Research
"""
    print(usage)


if __name__ == "__main__":
    main()
