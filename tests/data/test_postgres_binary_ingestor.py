import os
import secrets
import shutil
import subprocess

import pytest
from sqlalchemy import Column, Integer, MetaData, Table, Text, text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url
from sqlalchemy.schema import CreateSchema, DropSchema

from autorag_research.data.postgres_binary_ingestor import PostgresBinaryIngestor


def require_pg_tools():
    if not (shutil.which("pg_dump") and shutil.which("pg_restore")):
        pytest.skip("pg_dump and pg_restore are required for this test")


@pytest.fixture
def sample_schema(db_engine: Engine):
    schema = f"pg_ingest_{secrets.token_hex(4)}"
    with db_engine.begin() as conn:
        conn.execute(CreateSchema(schema))
    try:
        yield schema
    finally:
        with db_engine.begin() as conn:
            conn.execute(DropSchema(schema, cascade=True))


@pytest.fixture
def sample_table(db_engine: Engine, sample_schema: str):
    metadata = MetaData(schema=sample_schema)
    table = Table(
        "items",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", Text, nullable=False),
    )
    metadata.create_all(db_engine)
    with db_engine.begin() as conn:
        conn.execute(table.insert(), [
            {"id": 1, "name": "alpha"},
            {"id": 2, "name": "beta"},
        ])
    return table


def test_postgres_binary_ingestor_roundtrip(tmp_path, db_engine: Engine, sample_schema: str, sample_table: Table):
    require_pg_tools()

    url = make_url(os.environ["POSTGRES_URL"])
    dump_path = tmp_path / "sample.dump"

    PostgresBinaryIngestor(
        str(url),
        output_path=dump_path,
        pg_dump_args=[f"--schema={sample_schema}"],
    ).run()

    with db_engine.begin() as conn:
        conn.execute(text(f'DROP SCHEMA "{sample_schema}" CASCADE'))

    cmd = [
        "pg_restore",
        "--clean",
        "--if-exists",
        f"--dbname={url.database}",
        str(dump_path),
    ]
    if url.username:
        cmd.append(f"--username={url.username}")
    if url.host:
        cmd.append(f"--host={url.host}")
    if url.port:
        cmd.append(f"--port={url.port}")

    env = os.environ.copy()
    if url.password:
        env.setdefault("PGPASSWORD", url.password)

    subprocess.run(cmd, check=True, env=env)

    with db_engine.connect() as conn:
        rows = conn.execute(text(f'SELECT id, name FROM "{sample_schema}".items ORDER BY id')).fetchall()
    assert rows == [(1, "alpha"), (2, "beta")]

    with db_engine.begin() as conn:
        conn.execute(text(f'DROP SCHEMA "{sample_schema}" CASCADE'))
