"""Utility helpers for managing database dumps."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Sequence

from sqlalchemy import text
from sqlalchemy.engine import Engine


class DatabaseNotEmptyError(RuntimeError):
	"""Raised when attempting to restore into a non-empty database."""

def _ensure_empty(engine: Engine) -> None:
	with engine.connect() as conn:
		conn = conn.execution_options(isolation_level="AUTOCOMMIT")
		result = conn.execute(
			text(
				"""
				SELECT 1
				FROM pg_catalog.pg_class
				WHERE relkind IN ('r', 'm', 'f', 'p')
				  AND relnamespace NOT IN (
				    SELECT oid
				    FROM pg_catalog.pg_namespace
				    WHERE nspname IN ('pg_catalog', 'information_schema')
				  )
				LIMIT 1
				"""
			)
		)
		if result.first():
			raise DatabaseNotEmptyError("Target database must be empty before restore")


def restore_dump_into_database(
	*,
	engine: Engine,
	dump_path: str | os.PathLike[str],
	extra_args: Sequence[str] | None = None,
) -> None:
	"""Restore ``dump_path`` into the database referenced by ``engine``."""

	_dump = Path(dump_path)
	if not _dump.exists():
		raise FileNotFoundError(f"Dump file not found: {_dump}")

	_ensure_empty(engine)

	url = engine.url

	cmd = [
		"pg_restore",
		"--clean",
		"--if-exists",
		"--no-owner",
		"--no-privileges",
		"--dbname", url.database,
		str(_dump),
	]

	if url.username:
		cmd.extend(["--username", url.username])
	if url.host:
		cmd.extend(["--host", url.host])
	if url.port:
		cmd.extend(["--port", str(url.port)])

	if extra_args:
		cmd.extend(extra_args)

	env = os.environ.copy()
	if url.password:
		env.setdefault("PGPASSWORD", url.password)

	subprocess.run(cmd, check=True, env=env)
