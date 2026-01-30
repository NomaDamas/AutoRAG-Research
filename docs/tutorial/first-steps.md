# First Steps

Install and verify AutoRAG-Research.

## Install

```bash
pip install autorag-research
```

For development installation:

```bash
git clone https://github.com/NomaDamas/AutoRAG-Research.git
cd AutoRAG-Research
uv sync --all-extras
```

## Requirements

- Python 3.10+
- PostgreSQL 18 client tools (`pg_restore-18` for database restore)

Verify PostgreSQL tools:

```bash
pg_restore-18 --version
```

## Start PostgreSQL

```bash
docker-compose up -d
```

## Initialize Configuration

```bash
autorag-research init
```

Creates `configs/` directory with templates.

## Verify

```bash
autorag-research show ingestors
```

You should see available dataset ingestors:

| Name | Description |
|------|-------------|
| beir | BEIR benchmark datasets |
| mteb | MTEB retrieval tasks |
| ragbench | RAGBench benchmark |
| vidorev3 | ViDoRe V3 visual documents |

## Next

- [Text Retrieval](text-retrieval.md) - Run your first benchmark
- [Concepts](../learn/concepts.md) - Understand the system
