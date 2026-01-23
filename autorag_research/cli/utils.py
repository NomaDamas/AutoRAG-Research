"""CLI utility functions."""

import logging
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from platformdirs import user_data_dir
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

APP_NAME = "autorag-research"
CONFIG_REPO_URL = "https://raw.githubusercontent.com/vkehfdl1/AutoRAG-Research/main/configs"


def get_user_data_dir() -> Path:
    """Get user data directory for AutoRAG-Research."""
    return Path(user_data_dir(APP_NAME))


def get_config_dir() -> Path:
    """Get the configs directory in the current working directory."""
    return Path.cwd() / "configs"


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
    system_schemas = {"information_schema", "pg_catalog", "pg_toast", "public"}
    engine = create_engine(get_db_url(cfg))
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
