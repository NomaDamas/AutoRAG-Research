"""CLI utility functions."""

import logging
import os
import sys
from pathlib import Path

import yaml
from omegaconf import DictConfig, OmegaConf
from platformdirs import user_data_dir
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from autorag_research.cli.configs.db import DatabaseConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Config Discovery Functions
# =============================================================================


def discover_configs(config_dir: Path) -> dict[str, str]:
    """Scan YAML configs and return {name: description} dict.

    Args:
        config_dir: Directory containing YAML config files.

    Returns:
        Dictionary mapping config name (filename without .yaml) to description.
    """
    result = {}
    if not config_dir.exists():
        logger.warning(f"Config directory not found: {config_dir}")
        return result

    for yaml_file in sorted(config_dir.glob("*.yaml")):
        try:
            with open(yaml_file) as f:
                cfg = yaml.safe_load(f)
            name = yaml_file.stem
            # Use description if available, otherwise fallback to _target_ or filename
            description = cfg.get("description", cfg.get("_target_", "No description"))
            result[name] = description
        except Exception as e:
            logger.warning(f"Failed to parse {yaml_file}: {e}")
            result[yaml_file.stem] = "Error loading config"

    return result


def discover_pipelines() -> dict[str, str]:
    """Discover available pipelines from configs/pipelines/.

    Returns:
        Dictionary mapping pipeline name to description.
    """
    # Internal configs from working directory
    internal = discover_configs(get_config_dir() / "pipelines")
    # TODO (Phase 3): Add external plugin entry_points discovery here
    # external = _discover_plugin_configs("autorag_research.pipelines")
    # return {**internal, **external}
    return internal


def discover_metrics() -> dict[str, str]:
    """Discover available metrics from configs/metrics/.

    Returns:
        Dictionary mapping metric name to description.
    """
    # Internal configs from working directory
    internal = discover_configs(get_config_dir() / "metrics")
    # TODO (Phase 3): Add external plugin entry_points discovery here
    # external = _discover_plugin_configs("autorag_research.metrics")
    # return {**internal, **external}
    return internal


# =============================================================================
# Path and Config Utilities
# =============================================================================

APP_NAME = "autorag-research"
CONFIG_REPO_URL = "https://raw.githubusercontent.com/vkehfdl1/AutoRAG-Research/main/configs"


def get_user_data_dir() -> Path:
    """Get user data directory for AutoRAG-Research."""
    return Path(user_data_dir(APP_NAME))


def get_config_dir() -> Path:
    """Get the configs directory.

    Returns CONFIG_PATH if set by CLI, otherwise falls back to CWD/configs.
    """
    import autorag_research.cli as cli

    return cli.CONFIG_PATH or Path.cwd() / "configs"


def get_db_url(cfg: DictConfig) -> str:
    """Build PostgreSQL connection URL from config."""
    return f"postgresql+psycopg://{cfg.db.user}:{cfg.db.password}@{cfg.db.host}:{cfg.db.port}/{cfg.db.database}"


def create_session_factory(cfg: DictConfig) -> sessionmaker:
    """Create SQLAlchemy session factory from config."""
    engine = create_engine(get_db_url(cfg))
    return sessionmaker(bind=engine)


def print_config(cfg: DictConfig) -> None:
    """Print resolved configuration for debugging."""
    print(OmegaConf.to_yaml(cfg))


def list_schemas(cfg: DictConfig) -> list[str]:
    """List all user-created schemas in the database (excluding system schemas)."""
    return list_schemas_with_connection(
        host=cfg.db.host,
        port=cfg.db.port,
        user=cfg.db.user,
        password=cfg.db.password,
        database=cfg.db.database,
    )


def list_schemas_with_connection(host: str, port: int, user: str, password: str, database: str) -> list[str]:
    """List all user-created schemas in the database (excluding system schemas)."""
    system_schemas = {"information_schema", "pg_catalog", "pg_toast", "public"}
    db_url = f"postgresql+psycopg://{user}:{password}@{host}:{port}/{database}"
    engine = create_engine(db_url)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT schema_name FROM information_schema.schemata"))
        schemas = [row[0] for row in result if row[0] not in system_schemas]
    return sorted(schemas)


def get_schema_info(cfg: DictConfig, schema_name: str) -> dict:
    """Get detailed information about a schema."""
    engine = create_engine(get_db_url(cfg))
    tables_info: dict[str, dict] = {}
    info: dict = {"name": schema_name, "tables": tables_info}

    with engine.connect() as conn:
        # Get table names and row counts
        tables_result = conn.execute(
            text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = :schema
                AND table_type = 'BASE TABLE'
            """),
            {"schema": schema_name},
        )
        tables = [row[0] for row in tables_result]

        for table in tables:
            count_result = conn.execute(text(f'SELECT COUNT(*) FROM "{schema_name}"."{table}"'))  # noqa: S608
            count = count_result.scalar()
            tables_info[table] = {"row_count": count}

    return info


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_db_config_from_yaml() -> DatabaseConfig:
    """Load database config from configs/db/default.yaml if exists."""
    import autorag_research.cli as cli

    config_dir = cli.CONFIG_PATH or Path.cwd() / "configs"
    yaml_path = config_dir / "db" / "default.yaml"

    defaults = DatabaseConfig()

    if not yaml_path.exists():
        return defaults

    try:
        with open(yaml_path) as f:
            data = yaml.safe_load(f) or {}

        # Handle OmegaConf-style env var: ${oc.env:PGPASSWORD,postgres}
        password = data.get("password", defaults.password)
        if isinstance(password, str) and password.startswith("${"):
            password = os.environ.get("PGPASSWORD", "postgres")

        return DatabaseConfig(
            host=data.get("host", defaults.host),
            port=data.get("port", defaults.port),
            user=data.get("user", defaults.user),
            password=password,
            database=data.get("database", defaults.database),
        )
    except Exception as e:
        logger.warning(f"Failed to load DB config from YAML: {e}")
        return defaults
