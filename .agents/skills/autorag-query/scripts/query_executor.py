#!/usr/bin/env python3
"""
Safe SQL query executor for AutoRAG-Research database.

Loads connection from configs/db.yaml or environment variables,
validates queries for safety (SELECT-only), and executes with
timeout and result limits.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any

import typer
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from tabulate import tabulate

app = typer.Typer()


def load_db_connection(database: str | None = None) -> str:
    """Load database connection string using DBConnection class."""
    # Try loading from config file first
    config_path = Path.cwd() / "configs" / "db.yaml"

    if config_path.exists():
        # Import here to avoid issues if not in project root
        sys.path.insert(0, str(Path.cwd()))
        try:
            from autorag_research.orm.connection import DBConnection

            db_conn = DBConnection.from_config(Path.cwd() / "configs")
            if database:
                db_conn.database = database
        except Exception as e:
            typer.echo(f"Warning: Failed to load from config file: {e}", err=True)
            typer.echo("Falling back to environment variables...", err=True)
        else:
            return db_conn.db_url

    # Fallback to environment variables using DBConnection.from_env()
    try:
        sys.path.insert(0, str(Path.cwd()))
        from autorag_research.orm.connection import DBConnection

        db_conn = DBConnection.from_env()
        if database:
            db_conn.database = database
    except Exception as e:
        msg = (
            f"Failed to load database connection: {e}\n"
            "Either:\n"
            "1. Run from AutoRAG-Research project root with configs/db.yaml, or\n"
            "2. Set POSTGRES_* environment variables (POSTGRES_HOST, POSTGRES_PORT, "
            "POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB)"
        )
        typer.echo(msg, err=True)
        raise typer.Exit(code=1) from e
    else:
        return db_conn.db_url


def validate_query(query: str) -> None:
    """Validate query is SELECT-only and safe."""
    # Remove comments and normalize whitespace
    query_clean = re.sub(r"--.*$", "", query, flags=re.MULTILINE)
    query_clean = re.sub(r"/\*.*?\*/", "", query_clean, flags=re.DOTALL)
    query_clean = " ".join(query_clean.split()).upper()

    # Must contain SELECT
    if "SELECT" not in query_clean:
        msg = "Query must be a SELECT statement"
        raise ValueError(msg)

    # Reject DDL/DML keywords
    forbidden = [
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "CREATE",
        "ALTER",
        "TRUNCATE",
        "GRANT",
        "REVOKE",
        "EXECUTE",
        "CALL",
    ]
    for keyword in forbidden:
        if re.search(rf"\b{keyword}\b", query_clean):
            msg = f"Forbidden keyword: {keyword}. Only SELECT queries allowed."
            raise ValueError(msg)

    # Reject dangerous system functions
    dangerous_funcs = ["PG_READ_FILE", "PG_EXECUTE", "PG_LS_DIR", "COPY", "LO_IMPORT", "LO_EXPORT"]
    for func in dangerous_funcs:
        if func in query_clean:
            msg = f"Forbidden function: {func}"
            raise ValueError(msg)


def remove_vector_columns(query: str) -> str:
    """Remove vector/embedding columns from SELECT clause."""
    # Simple heuristic: remove problematic column names
    vector_cols = ["embedding", "embeddings", "bm25_tokens", "bm25vector"]

    for col in vector_cols:
        # Remove "column_name," or ", column_name"
        query = re.sub(rf"\b{col}\b\s*,", "", query, flags=re.IGNORECASE)
        query = re.sub(rf",\s*\b{col}\b", "", query, flags=re.IGNORECASE)

    return query


def execute_query(engine: Engine, query: str, timeout: int, limit: int) -> list[dict[str, Any]]:
    """Execute query with timeout and return results."""
    # Add LIMIT if not present
    if "LIMIT" not in query.upper() and limit > 0:
        query = f"{query.rstrip(';')} LIMIT {limit}"

    with engine.connect() as conn:
        # Set statement timeout
        conn.execute(text(f"SET statement_timeout = '{timeout}s'"))

        try:
            result = conn.execute(text(query))
            rows = result.fetchall()
            columns = result.keys()

            # Convert to list of dicts
            return [dict(zip(columns, row, strict=True)) for row in rows]

        except Exception as e:
            error_msg = str(e).lower()
            if "vector" in error_msg or "bm25" in error_msg:
                # Try removing vector columns
                typer.echo("Warning: Vector columns detected, retrying without them...", err=True)
                modified_query = remove_vector_columns(query)
                result = conn.execute(text(modified_query))
                rows = result.fetchall()
                columns = result.keys()
                return [dict(zip(columns, row, strict=True)) for row in rows]
            raise


def format_output(results: list[dict[str, Any]], output_format: str) -> str:
    """Format results as table, JSON, or CSV."""
    if not results:
        return "No results found."

    if output_format == "json":
        # Handle non-serializable types
        def json_serializer(obj):
            if hasattr(obj, "isoformat"):
                return obj.isoformat()
            return str(obj)

        return json.dumps(results, indent=2, default=json_serializer)

    elif output_format == "csv":
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
        rows = [[str(row[key]) if row[key] is not None else "NULL" for key in headers] for row in results]
        return tabulate(rows, headers=headers, tablefmt="simple")


@app.command()
def main(
    query: str = typer.Option(..., "--query", "-q", help="SQL query to execute (SELECT only)"),
    output_format: str = typer.Option("table", "--format", "-f", help="Output format: table, json, or csv"),
    timeout: int = typer.Option(10, "--timeout", "-t", help="Query timeout in seconds"),
    limit: int = typer.Option(10000, "--limit", "-l", help="Maximum rows to return (0=unlimited)"),
    database: str | None = typer.Option(None, "--database", "-d", help="Database name (overrides config/env)"),
):
    """Execute SELECT queries against AutoRAG-Research database."""
    if output_format not in ["table", "json", "csv"]:
        typer.echo(f"Error: Invalid format '{output_format}'. Choose: table, json, or csv", err=True)
        raise typer.Exit(code=1)

    try:
        # Validate query
        validate_query(query)

        # Load connection
        conn_string = load_db_connection(database)

        # Create engine
        engine = create_engine(conn_string)

        # Execute query
        results = execute_query(engine, query, timeout, limit)

        # Format and print output
        output = format_output(results, output_format)
        typer.echo(output)

        # Print row count to stderr
        typer.echo(f"\n({len(results)} rows)", err=True)

    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
