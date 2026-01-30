# autorag-research ingest

Ingest dataset into PostgreSQL.

## Synopsis

```bash
autorag-research ingest --name=<ingestor> [options]
```

## Options

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--name`, `-n` | Yes | - | Ingestor name |
| `--extra`, `-e` | No | - | Ingestor parameters (key=value) |
| `--subset` | No | `test` | Dataset split |
| `--embedding-model` | No | - | Embedding model config name |
| `--db-name` | No | auto | Database name |
| `--skip-embedding` | No | - | Skip embedding generation |
| `--embed-batch-size` | No | 128 | Embedding batch size |
| `--embed-concurrency` | No | 16 | Embedding concurrency |

## Examples

```bash
# Basic ingestion
autorag-research ingest --name=beir --extra dataset-name=scifact

# With custom embedding model
autorag-research ingest --name=beir --extra dataset-name=scifact --embedding-model=openai-small

# Multiple extra parameters
autorag-research ingest --name=ragbench --extra config=covidqa --extra split=test

# Data only (no embeddings)
autorag-research ingest --name=beir --extra dataset-name=scifact --skip-embedding

# Custom database name
autorag-research ingest --name=beir --extra dataset-name=scifact --db-name=my_custom_db
```

## Ingestor Parameters

Each ingestor accepts different `--extra` parameters:

| Ingestor | Parameters |
|----------|------------|
| beir | `dataset-name` |
| mteb | `dataset-name` |
| ragbench | `config` |
| mrtydi | `language` |
| vidorev3 | `dataset-name` |

## Related

- [show ingestors](show.md) - List available ingestors
- [data restore](data.md) - Download pre-indexed datasets
- [Custom Dataset Tutorial](../tutorial/custom-dataset.md) - Create your own
