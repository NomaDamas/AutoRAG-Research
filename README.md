# AutoRAG-Research

[![Release](https://img.shields.io/github/v/release/vkehfdl1/AutoRAG-Research)](https://img.shields.io/github/v/release/vkehfdl1/AutoRAG-Research)
[![Build status](https://img.shields.io/github/actions/workflow/status/vkehfdl1/AutoRAG-Research/main.yml?branch=main)](https://github.com/vkehfdl1/AutoRAG-Research/actions/workflows/main.yml?query=branch%3Amain)
[![Commit activity](https://img.shields.io/github/commit-activity/m/vkehfdl1/AutoRAG-Research)](https://img.shields.io/github/commit-activity/m/vkehfdl1/AutoRAG-Research)
[![License](https://img.shields.io/github/license/vkehfdl1/AutoRAG-Research)](https://img.shields.io/github/license/vkehfdl1/AutoRAG-Research)

Automate your RAG research.

- **Github repository**: <https://github.com/vkehfdl1/AutoRAG-Research/>
- **Documentation** <https://vkehfdl1.github.io/AutoRAG-Research/>

## CLI Usage

AutoRAG-Research provides a CLI tool for managing RAG research workflows.

### Installation

```bash
pip install autorag-research
```

or

```bash
uv pip install autorag-research
```

### Quick Start

```bash
# 1. Initialize configuration files
autorag-research init

# 2. Edit database settings
vim configs/db.yaml # OR your preferred editor

# 3. Ingest a dataset
autorag-research ingest --name beir --extra dataset-name=scifact

# 4. Run experiments
autorag-research run --db-name=beir_scifact_test
```

### Commands

#### `init` - Initialize Configuration Files

Downloads default configuration files to `./configs/` directory.

```bash
autorag-research init
```

This creates:
- `configs/db.yaml` - Database connection settings
- `configs/experiment.yaml` - Experiment configuration
- `configs/pipelines/**/*.yaml` - Pipeline configurations
- `configs/metrics/**/*.yaml` - Metric configurations

#### `ingest` - Ingest Datasets

Ingest datasets into PostgreSQL. Each ingestor supports different datasets.

```bash
# Show available ingestors
autorag-research ingest --help
```

```bash
autorag-research ingest --name beir --embedding-model mock --query-limit 5 --min-corpus-cnt 10 --extra dataset-name=scifact
```

#### `list` - List Available Resources

```bash
# List available ingestors
autorag-research list ingestors

# List available pipelines
autorag-research list pipelines

# List available metrics
autorag-research list metrics

# List database schemas
autorag-research list databases
```

#### `run` - Run Experiments

Run experiment pipelines with metrics evaluation. **Requires `--db-name` to specify the target database schema.**

```bash
# Basic run (uses configs/experiment.yaml)
autorag-research run --db-name=beir_scifact_test --verbose
```

### Environment Variables

| Variable              | Description |
|-----------------------|-------------|
| `POSTGRES_PASSWORD`   | PostgreSQL password (recommended for security) |
| `AUTORAG_CONFIG_PATH` | Default configuration directory path |


## Implementing New Pipelines (with Claude Code)

This project includes specialized Claude Code agents for implementing new RAG pipelines from research papers.

### Quick Start

```bash
# Full workflow from paper to validated code
/implement-pipeline https://arxiv.org/abs/2212.10496
```
