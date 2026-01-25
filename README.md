# AutoRAG-Research

[![Release](https://img.shields.io/github/v/release/vkehfdl1/AutoRAG-Research)](https://img.shields.io/github/v/release/vkehfdl1/AutoRAG-Research)
[![Build status](https://img.shields.io/github/actions/workflow/status/vkehfdl1/AutoRAG-Research/main.yml?branch=main)](https://github.com/vkehfdl1/AutoRAG-Research/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/vkehfdl1/AutoRAG-Research/branch/main/graph/badge.svg)](https://codecov.io/gh/vkehfdl1/AutoRAG-Research)
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
autorag-research init-config

# 2. Edit database settings
vi configs/db.yaml

# 3. Ingest a dataset
autorag-research ingest beir --dataset=scifact

# 4. Run experiments
autorag-research run --db-name=beir_scifact_test
```

### Commands

#### `init-config` - Initialize Configuration Files

Downloads default configuration files to `./configs/` directory.

```bash
autorag-research init-config
```

This creates:
- `configs/db.yaml` - Database connection settings
- `configs/experiment.yaml` - Experiment configuration
- `configs/pipelines/*.yaml` - Pipeline configurations
- `configs/metrics/*.yaml` - Metric configurations

#### `ingest` - Ingest Datasets

Ingest datasets into PostgreSQL. Each ingestor supports different datasets.

```bash
# Show available ingestors
autorag-research ingest --help

# Show available datasets for an ingestor
autorag-research ingest beir --list
```

**BEIR Benchmark:**
```bash
autorag-research ingest beir --dataset=scifact
autorag-research ingest beir --dataset=nfcorpus --subset=test --query-limit=100
```

**Mr. TyDi (Multilingual):**
```bash
autorag-research ingest mrtydi --language=english
autorag-research ingest mrtydi --language=korean --query-limit=500
```

**RAGBench:**
```bash
autorag-research ingest ragbench --configs=covidqa
autorag-research ingest ragbench --configs=covidqa,msmarco,hotpotqa
```

**MTEB Retrieval:**
```bash
autorag-research ingest mteb --task-name=NFCorpus
autorag-research ingest mteb --task-name=SciFact --score-threshold=2
```

**BRIGHT:**
```bash
autorag-research ingest bright --domains=biology,economics
autorag-research ingest bright --domains=stackoverflow --document-mode=long
```

**Common Options:**
| Option | Description |
|--------|-------------|
| `--subset` | Dataset split: train, dev, test (default: test) |
| `--query-limit` | Maximum number of queries to ingest |
| `--corpus-limit` | Maximum number of corpus documents |
| `--db-name` | Custom schema name (auto-generated if not specified) |

**Database Override Options:**
```bash
# Override database settings from configs/db.yaml
autorag-research ingest beir --dataset=scifact \
  --db-host=localhost \
  --db-port=5432 \
  --db-user=postgres \
  --db-password=secret \
  --db-database=mydb
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
autorag-research run --db-name=beir_scifact_test

# With additional options
autorag-research run --db-name=beir_scifact_test --max-retries=5 --verbose

# Use custom config file
autorag-research run --db-name=beir_scifact_test --config-name=my_experiment

# Use custom config directory
autorag-research --config-path=/my/configs run --db-name=beir_scifact_test

# Advanced: Hydra overrides as positional arguments
autorag-research run --db-name=test pipelines.0.k1=1.2
```

**Run Options:**
| Option | Short | Description |
|--------|-------|-------------|
| `--db-name` | `-d` | Database schema name (required) |
| `--config-name` | `-cn` | Config file name without .yaml (default: experiment) |
| `--max-retries` | | Maximum retries for failed operations |
| `--eval-batch-size` | | Batch size for evaluation |
| `--embedding-dim` | | Embedding dimension for vector operations |
| `--verbose` | `-v` | Enable verbose logging |

### Configuration

#### Database Configuration (`configs/db.yaml`)

```yaml
host: localhost
port: 5432
user: postgres
password: ${oc.env:PGPASSWORD,postgres}  # Uses PGPASSWORD env var or 'postgres'
database: autorag_research
```

#### Experiment Configuration (`configs/experiment.yaml`)

```yaml
defaults:
  - db: default
  - pipelines/bm25_baseline@pipelines.0
  - metrics/recall@metrics.0
  - metrics/ndcg@metrics.1
  - _self_

db_name: beir_scifact_test
embedding_dim: 1536
max_retries: 3
eval_batch_size: 100
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `PGPASSWORD` | PostgreSQL password (recommended for security) |
| `AUTORAG_CONFIG_PATH` | Default configuration directory path |


## Implementing New Pipelines (with Claude Code)

This project includes specialized Claude Code agents for implementing new RAG pipelines from research papers.

### Quick Start

```bash
# Full workflow from paper to validated code
/implement-pipeline https://arxiv.org/abs/2212.10496
```
