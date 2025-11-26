"""Database restoration utilities for AutoRAG-Research.

Provides functions for restoring PostgreSQL databases from dump files
created with pg_dump in custom format.
"""

import logging
import os
import subprocess
from pathlib import Path

import psycopg

logger = logging.getLogger("AutoRAG-Research")


# SQL to install vector extensions with fallback (vchord -> vectors -> vector)
_INSTALL_VECTOR_EXTENSIONS_SQL = """
DO $$
BEGIN
    BEGIN
        CREATE EXTENSION IF NOT EXISTS vchord CASCADE;
    EXCEPTION WHEN others THEN
        PERFORM 1;
    END;
    BEGIN
        CREATE EXTENSION IF NOT EXISTS vectors;
    EXCEPTION WHEN others THEN
        PERFORM 1;
    END;
    BEGIN
        CREATE EXTENSION IF NOT EXISTS vector;
    EXCEPTION WHEN others THEN
        PERFORM 1;
    END;
END $$;
"""


def _install_vector_extensions(
    host: str,
    user: str,
    password: str,
    database: str,
    port: int = 5432,
) -> None:
    """Install vector extensions (vchord, vectors, vector) with fallback.

    Tries to install extensions in order: vchord -> vectors -> vector.
    Silently continues if an extension is not available.

    Args:
        host: PostgreSQL server host.
        user: PostgreSQL user.
        password: User password.
        database: Target database name.
        port: PostgreSQL server port (default: 5432).
    """
    with psycopg.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        dbname=database,
    ) as conn:
        with conn.cursor() as cursor:
            cursor.execute(_INSTALL_VECTOR_EXTENSIONS_SQL)
        conn.commit()
    logger.info("Vector extensions installed successfully")


def restore_database(
    dump_file: str | Path,
    host: str,
    user: str,
    password: str,
    database: str,
    port: int = 5432,
    clean: bool = False,
    create: bool = False,
    no_owner: bool = False,
    install_extensions: bool = True,
    extra_args: list[str] | None = None,
) -> None:
    """Restore a PostgreSQL database from a dump file.

    Uses pg_restore to restore a database from a dump file created with
    pg_dump in custom format (--format=custom).

    Args:
        dump_file: Path to the dump file to restore from.
        host: PostgreSQL server host.
        user: PostgreSQL user with appropriate privileges.
        password: User password.
        database: Name of the target database to restore into.
        port: PostgreSQL server port (default: 5432).
        clean: If True, drop database objects before recreating them.
        create: If True, create the database before restoring.
        no_owner: If True, skip restoration of object ownership.
        install_extensions: If True, install vector extensions (vchord, vectors,
            vector) before restoring. Default is True.
        extra_args: Additional arguments to pass to pg_restore.

    Raises:
        FileNotFoundError: If the dump file does not exist.
        subprocess.CalledProcessError: If pg_restore fails.
        RuntimeError: If pg_restore command is not found.

    Example:
        >>> restore_database(
        ...     dump_file="/path/to/backup.dump",
        ...     host="localhost",
        ...     user="postgres",
        ...     password="mypassword",
        ...     database="autorag_research"
        ... )
    """
    dump_path = Path(dump_file)
    if not dump_path.exists():
        raise FileNotFoundError

    # Install vector extensions before restoring if requested
    if install_extensions:
        _install_vector_extensions(
            host=host,
            user=user,
            password=password,
            database=database,
            port=port,
        )

    cmd = [
        "pg_restore-18",
        f"--host={host}",
        f"--port={port}",
        f"--username={user}",
        f"--dbname={database}",
    ]

    if clean:
        cmd.append("--clean")
    if create:
        cmd.append("--create")
    if no_owner:
        cmd.append("--no-owner")

    if extra_args:
        cmd.extend(extra_args)

    cmd.append(str(dump_path))

    env = os.environ.copy()
    env["PGPASSWORD"] = password

    logger.info(f"Restoring database '{database}' from '{dump_path}'")

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
    except subprocess.CalledProcessError as e:
        logger.exception(f"pg_restore failed: {e.stderr}")
        raise

    logger.info(f"Database '{database}' restored successfully")
