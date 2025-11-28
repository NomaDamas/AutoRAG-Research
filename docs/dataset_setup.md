# Dataset Setup

This guide explains how to use the `setup_dataset` function to quickly set up pre-built datasets for your RAG research.

## Prerequisites

Before using `setup_dataset`, ensure you have to make sure the PostgresSQL 18 version and VectorChord extension is installed.
See [installation](./installation.md) for detailed instructions.

## Usage

### Basic Usage

```python
from autorag_research.data.util import setup_dataset

setup_dataset(
    dataset_name="scifact",
    embedding_model_name="embeddinggemma-300m",
    host="localhost",
    user="myuser",
    password="mypassword",
)
```

This will:

1. Download the pre-built dataset dump from AutoRAG-Research storage
2. Create a database named `scifact_embeddinggemma-300m`
3. Restore all tables and data to the database

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `dataset_name` | str | Yes | - | Name of the dataset (e.g., "scifact") |
| `embedding_model_name` | str | Yes | - | Name of the embedding model used |
| `host` | str | Yes | - | PostgreSQL server hostname |
| `user` | str | Yes | - | PostgreSQL username |
| `password` | str | Yes | - | PostgreSQL password |
| `port` | int | No | 5432 | PostgreSQL server port |
| `**kwargs` | - | No | - | Additional arguments passed to `restore_database` |

### Custom Port

If your PostgreSQL server is running on a different port:

```python
setup_dataset(
    dataset_name="scifact",
    embedding_model_name="embeddinggemma-300m",
    host="localhost",
    user="myuser",
    password="mypassword",
    port=5433,
)
```

## Available Datasets

Currently available dataset and embedding model combinations:

| Dataset | Embedding Model | Database Name |
|---------|-----------------|---------------|
| scifact | embeddinggemma-300m | scifact_embeddinggemma-300m |

More datasets will be added in future releases.

## Troubleshooting

### Connection Refused

If you see a connection error, ensure:

1. PostgreSQL server is running
2. The host and port are correct
3. The user has permission to create databases

### pg_restore Not Found

Ensure PostgreSQL 18 client tools are installed and `pg_restore-18` is in your PATH:

```bash
pg_restore-18 --version
```

### Database Already Exists

If the target database already exists, you may need to drop it first or use a different database name.
