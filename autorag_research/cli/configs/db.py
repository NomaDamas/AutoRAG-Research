"""Database configuration for CLI."""

from dataclasses import dataclass

from hydra.core.config_store import ConfigStore


@dataclass
class DatabaseConfig:
    """Database connection configuration."""

    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: str = "${oc.env:PGPASSWORD,postgres}"  # noqa: S105
    database: str = "autorag_research"


def register_db_configs() -> None:
    """Register database configs to the ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(group="db", name="default", node=DatabaseConfig)
