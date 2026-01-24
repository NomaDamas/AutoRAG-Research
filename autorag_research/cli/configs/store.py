"""Minimal ConfigStore registration for CLI commands."""

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore

from autorag_research.cli.configs.db import DatabaseConfig


@dataclass
class IngestConfig:
    """Configuration for the ingest command."""

    dataset: str
    embedding_model: str
    db: Any = field(default_factory=DatabaseConfig)
    embedding_dim: int = 1536
    skip_embedding: bool = False


@dataclass
class RunConfig:
    """Configuration for the run command.

    Pipelines and metrics are loaded from YAML via Hydra defaults.
    """

    db: Any = field(default_factory=DatabaseConfig)
    schema: str = ""
    pipelines: list = field(default_factory=list)
    metrics: list = field(default_factory=list)
    max_retries: int = 3
    eval_batch_size: int = 100


@dataclass
class ListConfig:
    """Configuration for the list command."""

    db: Any = field(default_factory=DatabaseConfig)
    resource: str = "datasets"


@dataclass
class InfoConfig:
    """Configuration for the info command."""

    db: Any = field(default_factory=DatabaseConfig)
    database: str = ""


def register_configs() -> None:
    """Register base configs to the Hydra ConfigStore."""
    cs = ConfigStore.instance()

    # Register command configs
    cs.store(name="ingest_config", node=IngestConfig)
    cs.store(name="run_config", node=RunConfig)
    cs.store(name="list_config", node=ListConfig)
    cs.store(name="info_config", node=InfoConfig)

    # Register database config group
    cs.store(group="db", name="default", node=DatabaseConfig)
