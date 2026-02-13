#!/usr/bin/env python3
"""
Safe SQL query executor for AutoRAG-Research database.

Loads connection from configs/db.yaml or environment variables,
validates queries for safety (SELECT-only), and executes with
timeout and result limits.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from tabulate import tabulate


def load_db_connection() -> str:
    """Load database connection string from config or environment."""
    # Try loading from config file first
    config_path = Path.cwd() / "configs" / "db.yaml"

    if config_path.exists():
        # Import here to avoid issues if not in project root
        sys.path.insert(0, str(Path.cwd()))
        try:
            from autorag_research.orm.connection import DBConnection
            db_conn = DBConnection.from_config(Path.cwd() / "configs")
            return db_conn.db_url
        except Exception as e:
            print(f"Warning: Failed to load from config file: {e}", file=sys.stderr)
            print("Falling back to environment variables...", file=sys.stderr)

    # Fallback to environment variables
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "")
    database = os.getenv("POSTGRES_DB", "autorag_research")

    if not password:
        raise ValueError(
            "Database connection not found. Either:\n"
            "1. Run from AutoRAG-Research project root with configs/db.yaml, or\n"
            "2. Set POSTGRES_* environment variables"
        )

    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{database}"


def validate_query(query: str) -> None:
    """Validate query is SELECT-only and safe."""
    # Remove comments and normalize whitespace
    query_clean = re.sub(r'--.*$', '', query, flags=re.MULTILINE)
    query_clean = re.sub(r'/\*.*?\*/', '', query_clean, flags=re.DOTALL)
    query_clean = ' '.join(query_clean.split()).upper()

    # Must contain SELECT
    if 'SELECT' not in query_clean:
        raise ValueError("Query must be a SELECT statement")

    # Reject DDL/DML keywords
    forbidden = [
        'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER',
        'TRUNCATE', 'GRANT', 'REVOKE', 'EXECUTE', 'CALL'
    ]
    for keyword in forbidden:
        if re.search(rf'\b{keyword}\b', query_clean):
            raise ValueError(f"Forbidden keyword: {keyword}. Only SELECT queries allowed.")

    # Reject dangerous system functions
    dangerous_funcs = [
        'PG_READ_FILE', 'PG_EXECUTE', 'PG_LS_DIR', 'COPY',
        'LO_IMPORT', 'LO_EXPORT'
    ]
    for func in dangerous_funcs:
        if func in query_clean:
            raise ValueError(f"Forbidden function: {func}")


def remove_vector_columns(query: str) -> str:
    """Remove vector/embedding columns from SELECT clause."""
    # Simple heuristic: remove problematic column names
    vector_cols = ['embedding', 'embeddings', 'bm25_tokens', 'bm25vector']

    for col in vector_cols:
        # Remove "column_name," or ", column_name"
        query = re.sub(rf'\b{col}\b\s*,', '', query, flags=re.IGNORECASE)
        query = re.sub(rf',\s*\b{col}\b', '', query, flags=re.IGNORECASE)

    return query


def execute_query(
    engine: Engine,
    query: str,
    timeout: int,
    limit: int
) -> list[dict[str, Any]]:
    """Execute query with timeout and return results."""
    # Add LIMIT if not present
    if 'LIMIT' not in query.upper() and limit > 0:
        query = f"{query.rstrip(';')} LIMIT {limit}"

    with engine.connect() as conn:
        # Set statement timeout
        conn.execute(text(f"SET statement_timeout = '{timeout}s'"))

        try:
            result = conn.execute(text(query))
            rows = result.fetchall()
            columns = result.keys()

            # Convert to list of dicts
            return [dict(zip(columns, row)) for row in rows]

        except Exception as e:
            error_msg = str(e).lower()
            if 'vector' in error_msg or 'bm25' in error_msg:
                # Try removing vector columns
                print("Warning: Vector columns detected, retrying without them...", file=sys.stderr)
                modified_query = remove_vector_columns(query)
                result = conn.execute(text(modified_query))
                rows = result.fetchall()
                columns = result.keys()
                return [dict(zip(columns, row)) for row in rows]
            raise


def format_output(results: list[dict[str, Any]], format: str) -> str:
    """Format results as table, JSON, or CSV."""
    if not results:
        return "No results found."

    if format == 'json':
        # Handle non-serializable types
        def json_serializer(obj):
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            return str(obj)

        return json.dumps(results, indent=2, default=json_serializer)

    elif format == 'csv':
        import csv
        import io

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        return output.getvalue()

    else:  # table
        # Convert dicts to list of lists
        headers = list(results[0].keys())
        rows = [[str(row[key]) if row[key] is not None else 'NULL' for key in headers] for row in results]
        return tabulate(rows, headers=headers, tablefmt='simple')


def main():
    parser = argparse.ArgumentParser(
        description='Execute SELECT queries against AutoRAG-Research database'
    )
    parser.add_argument(
        '--query', '-q',
        required=True,
        help='SQL query to execute (SELECT only)'
    )
    parser.add_argument(
        '--format', '-f',
        choices=['table', 'json', 'csv'],
        default='table',
        help='Output format (default: table)'
    )
    parser.add_argument(
        '--timeout', '-t',
        type=int,
        default=10,
        help='Query timeout in seconds (default: 10)'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=10000,
        help='Maximum rows to return (default: 10000, 0=unlimited)'
    )
    parser.add_argument(
        '--database', '-d',
        help='Database name (overrides config/env)'
    )

    args = parser.parse_args()

    try:
        # Validate query
        validate_query(args.query)

        # Load connection
        conn_string = load_db_connection()
        if args.database:
            # Replace database name in connection string
            conn_string = re.sub(r'/[^/]+$', f'/{args.database}', conn_string)

        # Create engine
        engine = create_engine(conn_string)

        # Execute query
        results = execute_query(engine, args.query, args.timeout, args.limit)

        # Format and print output
        output = format_output(results, args.format)
        print(output)

        # Print row count to stderr
        print(f"\n({len(results)} rows)", file=sys.stderr)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
