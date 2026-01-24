"""CLI utility functions."""

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from omegaconf import DictConfig, OmegaConf
from platformdirs import user_data_dir
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

if TYPE_CHECKING:
    from llama_index.core.base.embeddings.base import BaseEmbedding

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


@dataclass
class DatabaseConfig:
    """Database connection configuration."""

    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: str = "${oc.env:PGPASSWORD,postgres}"  # noqa: S105
    database: str = "autorag_research"


def load_db_config_from_yaml(
    host: str | None = None,
    port: int | None = None,
    user: str | None = None,
    password: str | None = None,
    database: str | None = None,
) -> DatabaseConfig:
    """
    Load database config from configs/db.yaml if exists.
    The parameters can be overridden via function arguments.
    """
    import autorag_research.cli as cli

    config_dir = cli.CONFIG_PATH or Path.cwd() / "configs"
    yaml_path = config_dir / "db.yaml"

    defaults = DatabaseConfig()

    if not yaml_path.exists():
        return defaults

    try:
        with open(yaml_path) as f:
            data = yaml.safe_load(f) or {}

        # Handle OmegaConf-style env var: ${oc.env:PGPASSWORD,postgres}
        loaded_password = data.get("password", defaults.password)
        if isinstance(loaded_password, str) and loaded_password.startswith("${"):
            loaded_password = os.environ.get("PGPASSWORD", "postgres")

        return DatabaseConfig(
            host=host or data.get("host", defaults.host),
            port=port or data.get("port", defaults.port),
            user=user or data.get("user", defaults.user),
            password=password or loaded_password,
            database=database or data.get("database", defaults.database),
        )
    except Exception as e:
        logger.warning(f"Failed to load DB config from YAML: {e}")
        return defaults


# =============================================================================
# Embedding Model Utilities
# =============================================================================


def load_embedding_model(config_name: str) -> "BaseEmbedding":
    """Load LlamaIndex embedding model directly from YAML via Hydra instantiate.

    Args:
        config_name: Name of the embedding config file (without .yaml extension).
                    e.g., "openai-small", "openai-large", "openai-like"

    Returns:
        LlamaIndex BaseEmbedding instance.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
    """
    from hydra.utils import instantiate
    from llama_index.core.base.embeddings.base import BaseEmbedding

    yaml_path = get_config_dir() / "embedding" / f"{config_name}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError

    cfg = OmegaConf.load(yaml_path)
    model = instantiate(cfg)

    if not isinstance(model, BaseEmbedding):
        raise TypeError(f"Expected BaseEmbedding, got {type(model)}")  # noqa: TRY003

    return model


def health_check_embedding(model: "BaseEmbedding") -> int:
    """Health check embedding model and return embedding dimension.

    Args:
        model: LlamaIndex BaseEmbedding instance.

    Returns:
        Embedding dimension (length of embedding vector).

    Raises:
        EmbeddingNotSetError: If health check fails.
    """
    from autorag_research.exceptions import EmbeddingNotSetError

    try:
        embedding = model.get_text_embedding("health check")
        return len(embedding)
    except Exception as e:
        raise EmbeddingNotSetError from e


def discover_embedding_configs() -> dict[str, str]:
    """Discover available embedding configs from configs/embedding/.

    Returns:
        Dictionary mapping config name to _target_ class path.
    """
    return discover_configs(get_config_dir() / "embedding")
