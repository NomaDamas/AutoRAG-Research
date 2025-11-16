"""Utility functions for database management in AutoRAG-Research.

Provides functions for database creation, dropping, and schema initialization
for PostgreSQL with vectorchord support.
"""

import logging

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT, connection

logger = logging.getLogger("AutoRAG-Research")


def _database_exists(conn: connection, database: str) -> bool:
    """Internal helper to check if a PostgreSQL database exists.

    Args:
        conn: Active psycopg2 connection.
        database: Name of the database to check.

    Returns:
        True if database exists, False otherwise.
    """
    cursor = None
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database,))
        exists = cursor.fetchone()
        return exists is not None
    finally:
        if cursor:
            cursor.close()


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
        psycopg2.Error: If connection or creation fails.

    Example:
        >>> create_database(
        ...     host="localhost",
        ...     user="postgres",
        ...     password="mypassword",
        ...     database="autorag_research"
        ... )
        True
    """
    conn = None
    cursor = None

    try:
        # Connect to default postgres database
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database="postgres",
        )

        # Set autocommit mode (CREATE DATABASE must run outside transaction)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

        # Check if database already exists using helper function
        if _database_exists(conn, database):
            logger.info(f"Database '{database}' already exists")
            return False

        # Create database with specified encoding and template
        # Use identifier quoting to prevent SQL injection
        cursor = conn.cursor()
        cursor.execute(
            f"CREATE DATABASE {psycopg2.extensions.quote_ident(database, cursor)} "
            f"ENCODING '{encoding}' "
            f"TEMPLATE {template}"
        )

        logger.info(f"Database '{database}' created successfully")
        return True

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


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
        psycopg2.Error: If connection or drop fails.

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
    conn = None
    cursor = None

    try:
        # Connect to default postgres database
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database="postgres",
        )

        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

        # Check if database exists using helper function
        if not _database_exists(conn, database):
            logger.info(f"Database '{database}' does not exist")
            return False

        # Terminate connections if force=True (PostgreSQL 13+)
        if force:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT pg_terminate_backend(pg_stat_activity.pid) "
                "FROM pg_stat_activity "
                "WHERE pg_stat_activity.datname = %s "
                "AND pid <> pg_backend_pid()",
                (database,),
            )

        # Drop database
        if not cursor:
            cursor = conn.cursor()
        cursor.execute(f"DROP DATABASE {psycopg2.extensions.quote_ident(database, cursor)}")

        logger.info(f"Database '{database}' dropped successfully")
        return True

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


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
    conn = None

    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database="postgres",
        )

        return _database_exists(conn, database)

    finally:
        if conn:
            conn.close()
