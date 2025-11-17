"""Utility functions for database management in AutoRAG-Research.

Provides functions for database creation, dropping, and schema initialization
for PostgreSQL with vectorchord support.
"""

import logging

import psycopg
from psycopg import Connection, sql

logger = logging.getLogger("AutoRAG-Research")


def _database_exists(conn: Connection, database: str) -> bool:
    """Internal helper to check if a PostgreSQL database exists.

    Args:
        conn: Active psycopg connection.
        database: Name of the database to check.

    Returns:
        True if database exists, False otherwise.
    """
    with conn.cursor() as cursor:
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database,))
        exists = cursor.fetchone()
        return exists is not None


def create_database(
    host: str,
    user: str,
    password: str,
    database: str,
    port: int = 5432,
    template: str = "template0",
    encoding: str = "UTF8",
) -> bool:
    """Create a new PostgreSQL database.

    Connects to the default 'postgres' database to create a new database.
    This operation requires CREATE DATABASE privileges.

    Args:
        host: PostgreSQL server host.
        user: PostgreSQL user with CREATE DATABASE privileges.
        password: User password.
        database: Name of the database to create.
        port: PostgreSQL server port (default: 5432).
        template: Template database to use (default: template0).
        encoding: Database encoding (default: UTF8).

    Returns:
        True if database was created, False if it already exists.

    Raises:
        psycopg.Error: If connection or creation fails.

    Example:
        >>> create_database(
        ...     host="localhost",
        ...     user="postgres",
        ...     password="mypassword",
        ...     database="autorag_research"
        ... )
        True
    """
    # Connect to default postgres database with autocommit
    # (CREATE DATABASE must run outside transaction)
    with psycopg.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        dbname="postgres",
        autocommit=True,
    ) as conn:
        # Check if database already exists using helper function
        if _database_exists(conn, database):
            logger.info(f"Database '{database}' already exists")
            return False

        # Create database with specified encoding and template
        # Use identifier quoting to prevent SQL injection
        with conn.cursor() as cursor:
            cursor.execute(
                sql.SQL("CREATE DATABASE {} ENCODING %s TEMPLATE {}").format(
                    sql.Identifier(database),
                    sql.Identifier(template),
                ),
                (encoding,),
            )

        logger.info(f"Database '{database}' created successfully")
        return True


def drop_database(
    host: str,
    user: str,
    password: str,
    database: str,
    port: int = 5432,
    force: bool = False,
) -> bool:
    """Drop a PostgreSQL database.

    Args:
        host: PostgreSQL server host.
        user: PostgreSQL user with DROP DATABASE privileges.
        password: User password.
        database: Name of the database to drop.
        port: PostgreSQL server port (default: 5432).
        force: If True, terminate all connections before dropping (PostgreSQL 13+).

    Returns:
        True if database was dropped, False if it didn't exist.

    Raises:
        psycopg.Error: If connection or drop fails.

    Example:
        >>> drop_database(
        ...     host="localhost",
        ...     user="postgres",
        ...     password="mypassword",
        ...     database="autorag_research",
        ...     force=True
        ... )
        True
    """
    # Connect to default postgres database with autocommit
    with psycopg.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        dbname="postgres",
        autocommit=True,
    ) as conn:
        # Check if database exists using helper function
        if not _database_exists(conn, database):
            logger.info(f"Database '{database}' does not exist")
            return False

        with conn.cursor() as cursor:
            # Terminate connections if force=True (PostgreSQL 13+)
            if force:
                cursor.execute(
                    "SELECT pg_terminate_backend(pg_stat_activity.pid) "
                    "FROM pg_stat_activity "
                    "WHERE pg_stat_activity.datname = %s "
                    "AND pid <> pg_backend_pid()",
                    (database,),
                )

            # Drop database
            cursor.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(database)))

        logger.info(f"Database '{database}' dropped successfully")
        return True


def database_exists(
    host: str,
    user: str,
    password: str,
    database: str,
    port: int = 5432,
) -> bool:
    """Check if a PostgreSQL database exists.

    Args:
        host: PostgreSQL server host.
        user: PostgreSQL user.
        password: User password.
        database: Name of the database to check.
        port: PostgreSQL server port (default: 5432).

    Returns:
        True if database exists, False otherwise.

    Example:
        >>> database_exists(
        ...     host="localhost",
        ...     user="postgres",
        ...     password="mypassword",
        ...     database="autorag_research"
        ... )
        True
    """
    with psycopg.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        dbname="postgres",
    ) as conn:
        return _database_exists(conn, database)
