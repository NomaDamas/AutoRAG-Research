# autorag-research data

Manage dataset dumps.

## Commands

| Command | Description |
|---------|-------------|
| `restore` | Download and restore dump to PostgreSQL |
| `dump` | Export database to dump file |
| `upload` | Upload dump to HuggingFace Hub |

## data restore

Download a pre-indexed dataset and restore to PostgreSQL.

### Synopsis

```bash
autorag-research data restore <ingestor> <filename> [options]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--db-name` | from filename | Target database name |
| `--clean` | false | Drop existing database first |
| `--no-owner` | false | Skip ownership restoration |

### Examples

```bash
# Basic restore
autorag-research data restore beir scifact_openai-small

# Custom database name
autorag-research data restore beir scifact_openai-small --db-name=my_scifact

# Clean restore (drop existing)
autorag-research data restore beir scifact_openai-small --clean
```

## data dump

Export a database to a dump file.

### Synopsis

```bash
autorag-research data dump [options]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--db-name` | required | Database to export |
| `--output`, `-o` | auto | Output file path |
| `--no-owner` | false | Skip ownership info |

### Examples

```bash
# Basic dump
autorag-research data dump --db-name=my_dataset_test_openai_small

# Custom output path
autorag-research data dump --db-name=my_dataset --output=./my_dump.sql
```

## data upload

Upload a dump file to HuggingFace Hub.

### Synopsis

```bash
autorag-research data upload <file> <ingestor> <filename> [options]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--message`, `-m` | auto | Commit message |

### Examples

```bash
# Upload to Hub
autorag-research data upload ./dump.sql my_dataset openai-small

# With custom message
autorag-research data upload ./dump.sql my_dataset openai-small -m "Initial upload"
```

## Related

- [ingest](ingest.md) - Ingest from source
- [show databases](show.md) - List databases
- [Custom Dataset Tutorial](../tutorial/custom-dataset.md) - Share your dataset
