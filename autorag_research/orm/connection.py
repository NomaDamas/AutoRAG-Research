import logging
import os
import subprocess
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

    def get_scoped_session_factory(self):
        """Create a thread-safe scoped SQLAlchemy session factory."""
        from sqlalchemy.orm import scoped_session

        return scoped_session(self.get_session_factory())

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
        self._create_bm25_indexes()
        self._run_migrations()
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
        self._run_migrations()
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

        self._create_bm25_indexes()
        self._run_migrations()

        logger.info(
            f"Database '{self.database}' created and vector extensions installed."
            "The BM25 indexes have been created and migrations have been run."
        )

    def _create_bm25_indexes(self):
        """Create BM25 indexes after tables exist."""
        if self.database is None:
            raise MissingDBNameError

        from autorag_research.orm.util import create_bm25_indexes

        create_bm25_indexes(
            host=self.host,
            port=self.port,
            user=self.username,
            password=self.password,
            database=self.database,
        )

    def _run_migrations(self):
        """Run schema migrations to add missing columns for backward compatibility."""
        if self.database is None:
            raise MissingDBNameError

        from autorag_research.orm.util import run_migrations

        run_migrations(
            host=self.host,
            port=self.port,
            user=self.username,
            password=self.password,
            database=self.database,
        )

    def terminate_connections(self):
        """Terminate all connections to this database except the current one.

        This is useful before dropping a database to ensure no active connections
        prevent the DROP DATABASE command from executing.
        """
        if self.database is None:
            raise MissingDBNameError

        # Connect to 'postgres' database to terminate connections
        admin_conn = DBConnection(
            host=self.host,
            port=self.port,
            username=self.username,
            password=self.password,
            database="postgres",
        )
        engine = admin_conn.get_engine()
        try:
            with engine.connect() as conn:
                conn.execute(
                    text(
                        """
                        SELECT pg_terminate_backend(pid)
                        FROM pg_stat_activity
                        WHERE datname = :dbname AND pid <> pg_backend_pid()
                        """
                    ),
                    {"dbname": self.database},
                )
                conn.commit()
            logger.info(f"Terminated all connections to database '{self.database}'.")
        finally:
            engine.dispose()

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

    def restore_database(
        self,
        dump_file: str | Path,
        clean: bool = False,
        create: bool = True,
        no_owner: bool = True,
        extra_args: list[str] | None = None,
    ) -> None:
        """Restore a PostgreSQL database from a dump file.

        Uses pg_restore to restore a database from a dump file created with
        pg_dump in custom format (--format=custom).

        Args:
            dump_file: Path to the dump file to restore from.
            clean: If True, drop database objects before recreating them.
            create: If True, create the database before restoring.
                Default is True.
            no_owner: If True, skip restoration of object ownership.
                Default is True.
            extra_args: Additional arguments to pass to pg_restore.

        Raises:
            MissingDBNameError: If database name is not set.
            FileNotFoundError: If the dump file does not exist.
            subprocess.CalledProcessError: If pg_restore fails.
            RuntimeError: If pg_restore command is not found.
        """
        if self.database is None:
            raise MissingDBNameError

        dump_path = Path(dump_file)
        if not dump_path.exists():
            raise FileNotFoundError

        if create:
            self.create_database()

        optional_flags = [
            ("--clean", clean),
            ("--no-owner", no_owner),
        ]
        cmd = [
            "pg_restore",
            f"--host={self.host}",
            f"--port={self.port}",
            f"--username={self.username}",
            f"--dbname={self.database}",
            *[flag for flag, enabled in optional_flags if enabled],
            *(extra_args or []),
            str(dump_path),
        ]

        env = os.environ.copy()
        env["PGPASSWORD"] = self.password

        logger.info(f"Restoring database '{self.database}' from '{dump_path}'")

        try:
            result = subprocess.run(  # noqa: S603
                cmd,
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            if result.stdout:
                logger.debug(result.stdout)
        except FileNotFoundError as e:
            msg = "pg_restore command not found. Ensure PostgreSQL client tools are installed."
            raise RuntimeError(msg) from e
        except subprocess.CalledProcessError as e:
            logger.exception(f"pg_restore failed: {e.stderr}")
            raise

        self._create_bm25_indexes()
        self._run_migrations()
        logger.info(f"Database '{self.database}' restored successfully")

    def dump_database(
        self,
        output_file: str | Path,
        output_format: str = "custom",
        no_owner: bool = True,
        extra_args: list[str] | None = None,
    ) -> Path:
        """Dump the database to a file.

        Uses pg_dump to create a database dump file that can be restored
        with pg_restore or the restore_database function.

        Args:
            output_file: Path to the output dump file.
            output_format: Output format - "custom" (default), "plain", "directory", or "tar".
            no_owner: If True, skip output of commands to set ownership.
                Default is True.
            extra_args: Additional arguments to pass to pg_dump.

        Returns:
            Path to the created dump file.

        Raises:
            MissingDBNameError: If database name is not set.
            subprocess.CalledProcessError: If pg_dump fails.
            RuntimeError: If pg_dump command is not found.
        """
        if self.database is None:
            raise MissingDBNameError

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "pg_dump",
            f"--host={self.host}",
            f"--port={self.port}",
            f"--username={self.username}",
            f"--dbname={self.database}",
            f"--format={output_format}",
            f"--file={output_path}",
        ]

        if no_owner:
            cmd.append("--no-owner")

        if extra_args:
            cmd.extend(extra_args)

        env = os.environ.copy()
        env["PGPASSWORD"] = self.password

        logger.info(f"Dumping database '{self.database}' to '{output_path}'")

        try:
            result = subprocess.run(  # noqa: S603
                cmd,
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            if result.stdout:
                logger.debug(result.stdout)
        except FileNotFoundError as e:
            msg = "pg_dump command not found. Ensure PostgreSQL client tools are installed."
            raise RuntimeError(msg) from e
        except subprocess.CalledProcessError as e:
            logger.exception(f"pg_dump failed: {e.stderr}")
            raise

        if not output_path.exists():
            msg = f"Dump file was not created: {output_path}"
            raise RuntimeError(msg)

        logger.info(f"Database '{self.database}' dumped successfully to '{output_path}'")
        return output_path

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
            username=str(username),
            password=str(password),
            database=database,
        )
