"""Utilities for exporting PostgreSQL datasets via `pg_dump`."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from sqlalchemy.engine.url import make_url

from autorag_research.data.base import DataIngestor


class PostgresBinaryIngestor(DataIngestor):
    """Export a PostgreSQL database using native `pg_dump`."""

    def __init__(
        self,
        dsn: str,
        *,
        output_path: str | os.PathLike[str],
        pg_dump_args: list[str] | None = None,
    ) -> None:
        super().__init__(dsn, schema=None)
        self.output_path = Path(output_path)
        self.pg_dump_args = list(pg_dump_args or [])
        self.url = make_url(dsn)

        if self.url.drivername == "postgresql+psycopg2":
            raise ValueError(
                "PostgresBinaryIngestor only supports psycopg (v3). Use a DSN with the "
                "'postgresql+psycopg' driver or omit the driver for SQLAlchemy's default."
            )
        if self.url.drivername not in {
            "postgresql",
            "postgresql+psycopg",
        }:
            raise ValueError(
                "PostgresBinaryIngestor requires a PostgreSQL DSN using the psycopg driver."
            )

    def prepare(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def ingest(self) -> None:
        cmd: list[str] = [
            "pg_dump",
            "--format=custom",
            f"--file={self.output_path}",
        ]

        if self.url.database:
            cmd.append(f"--dbname={self.url.database}")
        if self.url.username:
            cmd.append(f"--username={self.url.username}")
        if self.url.host:
            cmd.append(f"--host={self.url.host}")
        if self.url.port:
            cmd.append(f"--port={self.url.port}")

        cmd.extend(self.pg_dump_args)

        env = os.environ.copy()
        if self.url.password:
            env.setdefault("PGPASSWORD", self.url.password)

        subprocess.run(cmd, check=True, env=env)

    def finalize(self) -> None:
        pass

    @classmethod
    def ingest_to_file(
        cls,
        dsn: str,
        *,
        output_path: str | os.PathLike[str],
        pg_dump_args: list[str] | None = None,
        config_override: dict[str, Any] | None = None,
    ) -> Path:
        config = {"pg_dump_args": pg_dump_args or []}
        if config_override:
            config.update(config_override)

        ingestor = cls(dsn, output_path=output_path, **config)
        ingestor.run()
        return ingestor.output_path


__all__ = ["PostgresBinaryIngestor"]
