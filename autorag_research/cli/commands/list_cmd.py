"""list command - List available resources."""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from autorag_research.cli.configs.datasets import AVAILABLE_DATASETS
from autorag_research.cli.configs.metrics import AVAILABLE_METRICS
from autorag_research.cli.configs.pipelines import AVAILABLE_PIPELINES
from autorag_research.cli.utils import list_schemas

CONFIG_PATH = str(Path.cwd() / "configs")


@hydra.main(version_base=None, config_name="list_config", config_path=None)
def list_resources(cfg: DictConfig) -> None:
    """List available resources based on the resource type."""
    resource = cfg.get("resource", "datasets")

    if resource == "datasets":
        print_datasets()
    elif resource == "pipelines":
        print_pipelines()
    elif resource == "metrics":
        print_metrics()
    elif resource == "databases":
        print_databases(cfg)
    else:
        print(f"Unknown resource type: {resource}")
        print("Available types: datasets, pipelines, metrics, databases")


def print_datasets() -> None:
    """Print available datasets."""
    print("\nAvailable Datasets:")
    print("-" * 60)
    for name, description in sorted(AVAILABLE_DATASETS.items()):
        print(f"  {name:<25} {description}")
    print("\nUsage: autorag-research ingest dataset=<name>")


def print_pipelines() -> None:
    """Print available pipelines."""
    print("\nAvailable Pipelines:")
    print("-" * 60)
    for name, description in sorted(AVAILABLE_PIPELINES.items()):
        print(f"  {name:<20} {description}")
    print("\nSee configs/pipelines/ for configuration options")


def print_metrics() -> None:
    """Print available metrics."""
    print("\nAvailable Metrics:")
    print("-" * 60)
    for name, description in sorted(AVAILABLE_METRICS.items()):
        print(f"  {name:<15} {description}")
    print("\nSee configs/metrics/ for configuration options")


def print_databases(cfg: DictConfig) -> None:
    """Print database schemas."""
    print("\nDatabase Schemas:")
    print("-" * 60)
    try:
        schemas = list_schemas(cfg)
        if schemas:
            for schema in schemas:
                print(f"  {schema}")
        else:
            print("  No user schemas found.")
        print(f"\nDatabase: {cfg.db.host}:{cfg.db.port}/{cfg.db.database}")
        print("\nUsage: autorag-research info database=<schema_name>")
    except Exception as e:
        print(f"  Error connecting to database: {e}")
        print("  Make sure PostgreSQL is running and credentials are correct.")
