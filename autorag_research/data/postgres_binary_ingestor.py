"""Utilities for exporting PostgreSQL datasets to compressed binary archives."""

from __future__ import annotations

import gzip
import os
import pickle
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine, create_engine


DEFAULT_TABLES = (
    "file",
    "document",
    "page",
    "caption",
    "chunk",
    "image_chunk",
    "caption_chunk_relation",
    "query",
    "retrieval_relation",
    "pipeline",
    "metric",
    "experiment_result",
    "image_chunk_retrieved_result",
    "chunk_retrieved_result",
    "summary",
)


class PostgresBinaryIngestor:
    """Serialize PostgreSQL tables into a single compressed binary asset."""

    def __init__(
        self,
        dsn: str,
        *,
        output_path: str | os.PathLike[str],
        schema: str | None = None,
        table_names: tuple[str, ...] = DEFAULT_TABLES,
        chunk_size: int = 1000,
    ) -> None:
        super().__init__(dsn, schema=schema)
        self.output_path = Path(output_path)
        self.table_names = table_names
        self.chunk_size = chunk_size
        self._engine: Engine | None = None

    # Lifecycle -----------------------------------------------------------------

    def prepare(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_engine(self.dsn, future=True)

    def ingest(self) -> None:
        if self._engine is None:
            raise RuntimeError("Engine not initialized; call prepare() before ingest().")

        payload: dict[str, list[dict[str, Any]]] = {}
        metadata: dict[str, Any] = {
            "schema": self.schema,
            "tables": list(self.table_names),
        }

        with self._engine.connect() as conn:
            if self.schema:
                conn.execute(text(f"SET search_path TO {self.schema}"))

            for table in self.table_names:
                rows: list[dict[str, Any]] = []
                offset = 0
                while True:
                    result = conn.execute(
                        text(f"SELECT * FROM {table} OFFSET :offset LIMIT :limit"),
                        {"offset": offset, "limit": self.chunk_size},
                    )
                    chunk = [dict(row._mapping) for row in result]
                    rows.extend(chunk)
                    if len(chunk) < self.chunk_size:
                        break
                    offset += self.chunk_size

                payload[table] = rows

        binary_blob = pickle.dumps({"metadata": metadata, "payload": payload})
        with gzip.open(self.output_path, "wb") as fp:
            fp.write(binary_blob)

    # Public helpers ------------------------------------------------------------

    @classmethod
    def ingest_to_file(
        cls,
        dsn: str,
        *,
        output_path: str | os.PathLike[str],
        schema: str | None = None,
        table_names: tuple[str, ...] = DEFAULT_TABLES,
        chunk_size: int = 1000,
        config_override: dict[str, Any] | None = None,
    ) -> Path:
        config_override = config_override or {}
        ingestor = cls(
            dsn,
            output_path=output_path,
            schema=schema,
            table_names=table_names,
            chunk_size=chunk_size,
        )
        ingestor.configure(**config_override)
        ingestor.run()
        return Path(output_path)


def load_binary_archive(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Utility to hydrate previously exported archives."""

    with gzip.open(path, "rb") as fp:
        return pickle.load(fp)

