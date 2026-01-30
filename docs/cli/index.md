# CLI Reference

Command-line interface for AutoRAG-Research.

## Commands

| Command | Description |
|---------|-------------|
| [init](init.md) | Download config templates |
| [ingest](ingest.md) | Ingest dataset |
| [run](run.md) | Execute pipelines |
| [show](show.md) | List resources |
| [data](data.md) | Manage datasets |

## Global Options

| Option | Description |
|--------|-------------|
| `--config-path`, `-cp` | Config directory (default: `./configs`) |
| `--version`, `-V` | Show version |
| `--help` | Show help |

## Environment Variables

| Variable | Description |
|----------|-------------|
| POSTGRES_HOST | PostgreSQL host |
| POSTGRES_PORT | PostgreSQL port |
| POSTGRES_USER | PostgreSQL user |
| POSTGRES_PASSWORD | PostgreSQL password |
