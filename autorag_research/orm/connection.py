import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from sqlalchemy import Engine, text

from autorag_research.exceptions import MissingDBNameError

logger = logging.getLogger("AutoRAG-Research")


@dataclass
class DBConnection:
    """Database connection configuration."""

    host: str
    port: int
    username: str
    password: str
    database: str | None = None

    @property
    def db_url(self) -> str:
        """Construct the SQLAlchemy database URL."""
        if self.database is None:
            return f"postgresql+psycopg://{self.username}:{self.password}@{self.host}:{self.port}"
        return f"postgresql+psycopg://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

    @property
    def duckdb_url(self) -> str:
        """Construct the DuckDB-compatible PostgreSQL URL (without driver specification)."""
        if self.database is None:
            return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}"
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

    def get_engine(self) -> Engine:
        """Create a SQLAlchemy engine using the connection configuration."""
        from sqlalchemy import create_engine

        return create_engine(self.db_url)

    def get_session_factory(self):
        """Create a SQLAlchemy session factory using the connection configuration."""
        from sqlalchemy.orm import sessionmaker

        engine = self.get_engine()
        return sessionmaker(bind=engine)

    def detect_primary_key_type(self, schema_name: str = "public") -> Literal["bigint", "string"]:
        engine = self.get_engine()

        query = text("""
                     SELECT t.typname
                     FROM pg_attribute a
                              JOIN pg_class c ON a.attrelid = c.oid
                              JOIN pg_namespace n ON c.relnamespace = n.oid
                              JOIN pg_type t ON a.atttypid = t.oid
                     WHERE n.nspname = :schema
                       AND c.relname = 'query'
                       AND a.attname = 'id'
                       AND a.attnum > 0;
                     """)

        try:
            with engine.connect() as conn:
                result = conn.execute(query, {"schema": schema_name})
                row = result.fetchone()
                if row is None:
                    raise ValueError(f"Primary key type not found in {schema_name}.query")  # noqa: TRY003
                typname = row[0].lower()

                if typname in ("int2", "int4", "int8", "integer", "bigint", "serial", "bigserial"):
                    logger.info(f"Detected primary key type: int (pg type: {typname})")
                    return "bigint"

                if typname in ("text", "varchar", "char", "bpchar", "name", "uuid"):
                    logger.info(f"Detected primary key type: str (pg type: {typname})")
                    return "string"

                raise ValueError(f"Unknown primary key type: {typname}")  # noqa: TRY003
        finally:
            engine.dispose()

    def detect_embedding_dimension(self, schema_name: str = "public") -> int:
        engine = self.get_engine()

        query = text("""
                     SELECT a.atttypmod AS dimension
                     FROM pg_attribute a
                              JOIN pg_class c ON a.attrelid = c.oid
                              JOIN pg_namespace n ON c.relnamespace = n.oid
                     WHERE n.nspname = :schema
                       AND c.relname = 'chunk'
                       AND a.attname = 'embedding'
                     """)

        try:
            with engine.connect() as conn:
                result = conn.execute(query, {"schema": schema_name})
                row = result.fetchone()
                if row and row[0] and row[0] > 0:
                    dimension = row[0]
                    logger.info(f"Auto-detected embedding dimension: {dimension} from schema '{schema_name}'")
                    return dimension
                raise ValueError(f"Embedding dimension not found in {schema_name}.chunk")  # noqa: TRY003
        finally:
            engine.dispose()

    def get_database_names(self) -> list[str]:
        engine = self.get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT datname FROM pg_database WHERE datistemplate = false ORDER BY datname"))
            databases = [row[0] for row in result]
        return databases

    def create_schema(self, embedding_dim: int, primary_key_type: Literal["bigint", "string"]):
        from autorag_research.orm.schema_factory import create_schema

        schema = create_schema(embedding_dim, primary_key_type)
        schema.Base.metadata.create_all(self.get_engine())
        return schema

    def get_schema(self, schema_name: str = "public"):
        """Auto-detect embedding dimension and primary key type, then create schema.

        Args:
            schema_name: Database schema name to detect types from.

        Returns:
            Schema object with auto-detected embedding dimension and primary key type.
        """
        from autorag_research.orm.schema_factory import create_schema

        embedding_dim = self.detect_embedding_dimension(schema_name)
        pkey_type = self.detect_primary_key_type(schema_name)
        return create_schema(embedding_dim, pkey_type)

    def create_database(self):
        if self.database is None:
            raise MissingDBNameError

        from autorag_research.orm.util import create_database, install_vector_extensions

        create_database(
            host=self.host,
            port=self.port,
            user=self.username,
            password=self.password,
            database=self.database,
        )

        install_vector_extensions(
            host=self.host,
            port=self.port,
            user=self.username,
            password=self.password,
            database=self.database,
        )

        logger.info(f"Database '{self.database}' created and vector extensions installed.")

    def drop_database(self):
        if self.database is None:
            raise MissingDBNameError

        from autorag_research.orm.util import drop_database

        drop_database(
            host=self.host,
            port=self.port,
            user=self.username,
            password=self.password,
            database=self.database,
        )

        logger.info(f"Database '{self.database}' dropped.")

    @classmethod
    def from_config(cls, config_path: Path | None = None) -> "DBConnection":
        """Load database connection configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file. If None, uses default path.
        Returns:
            DBConnection instance with loaded configuration.
        """
        from omegaconf import DictConfig, OmegaConf

        from autorag_research import cli

        resolved_path = config_path or cli.CONFIG_PATH
        if resolved_path is None:
            raise ValueError("Config path not provided and CONFIG_PATH is not set.")  # noqa: TRY003

        cfg = OmegaConf.load(resolved_path / "db.yaml")
        if not isinstance(cfg, DictConfig):
            raise TypeError("db.yaml must be a YAML mapping.")  # noqa: TRY003

        password = os.environ.get("POSTGRES_PASSWORD", cfg.get("password"))
        if password is None:
            raise ValueError("Database password not found in config or POSTGRES_PASSWORD env variable.")  # noqa: TRY003

        return cls(
            host=cfg.host,
            port=cfg.port,
            username=cfg.user,
            password=password,
            database=cfg.get("database"),
        )

    @classmethod
    def from_env(cls) -> "DBConnection":
        """Load database connection configuration from environment variables.

        Returns:
            DBConnection instance with loaded configuration.
        """
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = int(os.getenv("POSTGRES_PORT", "5432"))
        username = os.getenv("POSTGRES_USER")
        password = os.getenv("POSTGRES_PASSWORD")
        database = os.getenv("POSTGRES_DB", None)

        if not all([host, port, username, password]):
            raise ValueError("Missing required database environment variables.")  # noqa: TRY003

        return cls(
            host=host,
            port=port,
            username=username,
            password=password,
            database=database,
        )
