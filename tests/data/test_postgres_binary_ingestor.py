"""Integration test for PostgresBinaryIngestor using dedicated databases."""

from __future__ import annotations

import os
import secrets
import shutil
import subprocess

import pytest
from sqlalchemy import Column, Integer, MetaData, Table, Text, create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import URL, make_url

from autorag_research.data.postgres_binary_ingestor import PostgresBinaryIngestor

TEST_DSN_ENV = "POSTGRES_URL_TEST_INGEST"
DEFAULT_ADMIN_DB = "postgres"


def _ensure_tools_available() -> None:
    if not (shutil.which("pg_dump") and shutil.which("pg_restore")):
        pytest.skip("pg_dump and pg_restore are required for this test")


def _get_base_url() -> URL:
    dsn = os.getenv(TEST_DSN_ENV)
    if not dsn:
        pytest.skip(f"{TEST_DSN_ENV} is not configured")
    return make_url(dsn)


def _admin_url(base: URL) -> URL:
    admin_db = base.database or DEFAULT_ADMIN_DB
    return base.set(database=admin_db)


def _execute_ddl(engine: Engine, statement: str) -> None:
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text(statement))


def _drop_database(engine: Engine, database: str) -> None:
    _execute_ddl(engine, f'DROP DATABASE IF EXISTS "{database}"')


def _create_database(engine: Engine, database: str) -> None:
    _execute_ddl(engine, f'CREATE DATABASE "{database}"')


@pytest.fixture()
def source_engine() -> Engine:
    _ensure_tools_available()
    base_url = _get_base_url()
    admin_engine = create_engine(_admin_url(base_url))

    db_name = f"ingest_src_{secrets.token_hex(4)}"
    _drop_database(admin_engine, db_name)
    _create_database(admin_engine, db_name)

    engine = create_engine(base_url.set(database=db_name), pool_pre_ping=True)
    try:
        yield engine
    finally:
        engine.dispose()
        _drop_database(admin_engine, db_name)
        admin_engine.dispose()


@pytest.fixture()
def scratch_schema(source_engine: Engine) -> str:
    schema = f"pg_ingest_{secrets.token_hex(4)}"
    with source_engine.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA "{schema}"'))
    try:
        yield schema
    finally:
        with source_engine.begin() as conn:
            conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))


@pytest.fixture()
def sample_table(source_engine: Engine, scratch_schema: str) -> Table:
    metadata = MetaData(schema=scratch_schema)
    table = Table(
        "items",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", Text, nullable=False),
    )
    metadata.create_all(source_engine)
    with source_engine.begin() as conn:
        conn.execute(table.insert(), [
            {"id": 1, "name": "alpha"},
            {"id": 2, "name": "beta"},
        ])
    return table


def test_postgres_binary_ingestor_roundtrip(tmp_path, source_engine: Engine, scratch_schema: str, sample_table: Table):
    base_url = _get_base_url()
    dump_path = tmp_path / "sample.dump"

    PostgresBinaryIngestor(
        str(base_url.set(database=source_engine.url.database)),
        output_path=dump_path,
        pg_dump_args=[f"--schema={scratch_schema}"],
    ).run()

    target_db = f"ingest_restore_{secrets.token_hex(4)}"
    admin_engine = create_engine(_admin_url(base_url))
    _drop_database(admin_engine, target_db)
    _create_database(admin_engine, target_db)
    admin_engine.dispose()

    restore_cmd = [
        "pg_restore",
        "--clean",
        "--if-exists",
        f"--dbname={target_db}",
        str(dump_path),
    ]
    if base_url.username:
        restore_cmd.append(f"--username={base_url.username}")
    if base_url.host:
        restore_cmd.append(f"--host={base_url.host}")
    if base_url.port:
        restore_cmd.append(f"--port={base_url.port}")

    env = os.environ.copy()
    if base_url.password:
        env.setdefault("PGPASSWORD", base_url.password)

    subprocess.run(restore_cmd, check=True, env=env)

    restore_engine = create_engine(base_url.set(database=target_db), pool_pre_ping=True)
    try:
        with restore_engine.connect() as conn:
            rows = conn.execute(text(f'SELECT id, name FROM "{scratch_schema}".items ORDER BY id')).fetchall()
        assert rows == [(1, "alpha"), (2, "beta")]
    finally:
        restore_engine.dispose()
        admin_engine = create_engine(_admin_url(base_url))
        _drop_database(admin_engine, target_db)
        admin_engine.dispose()
