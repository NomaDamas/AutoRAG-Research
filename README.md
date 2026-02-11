![Thumbnail](./assets/thumbnail.png)

# AutoRAG-Research

<br /> Automate your RAG research <br />

---

 [Documentation](https://nomadamas.github.io/AutoRAG-Research/)

---

## What is AutoRAG-Research?

| Problem | What AutoRAG-Research does |
|---------|---------------------------|
| Every dataset has a different format. | We unify the formats and pre-compute embeddings for you. Just download and use. |
| Comparing against SOTA pipelines requires implementing each one. | We implement SOTA pipelines from papers. Benchmark yours against them. |
| Every paper claims SOTA. Which one actually is? | Run all pipelines on your data with one command and compare. |

Which pipeline is really SOTA? What datasets are out there? Find it all here.


(Action Video)


## Available Datasets

We provide pre-processed datasets with unified formats. Some include pre-computed embeddings.

**Text**

| Dataset | Description |
|---------|-------------|
| [BEIR](https://vkehfdl1.github.io/AutoRAG-Research/datasets/text/beir/) | Standard IR benchmark across 14 diverse domains (scifact, nq, hotpotqa, ...) |
| [MTEB](https://vkehfdl1.github.io/AutoRAG-Research/datasets/text/mteb/) | Large-scale embedding benchmark with any MTEB retrieval task |
| [RAGBench](https://vkehfdl1.github.io/AutoRAG-Research/datasets/text/ragbench/) | End-to-end RAG evaluation with generation ground truth across 12 domains |
| [MrTyDi](https://vkehfdl1.github.io/AutoRAG-Research/datasets/text/mrtydi/) | Multilingual retrieval across 11 languages |
| [BRIGHT](https://vkehfdl1.github.io/AutoRAG-Research/datasets/text/bright/) | Reasoning-intensive retrieval requiring multi-step inference |

**Image**

| Dataset | Description |
|---------|-------------|
| [ViDoRe](https://vkehfdl1.github.io/AutoRAG-Research/datasets/multimodal/vidore/) | Visual document QA with 1:1 query-to-page mapping |
| [ViDoRe v2](https://vkehfdl1.github.io/AutoRAG-Research/datasets/multimodal/vidorev2/) | Visual document retrieval with corpus-level search |
| [ViDoRe v3](https://vkehfdl1.github.io/AutoRAG-Research/datasets/multimodal/vidorev3/) | Visual document retrieval across 8 industry domains |
| [VisRAG](https://vkehfdl1.github.io/AutoRAG-Research/datasets/multimodal/visrag/) | Vision-based RAG benchmark (ChartQA, SlideVQA, DocVQA, ...) |

**Text + Image**

| Dataset | Description |
|---------|-------------|
| [Open-RAGBench](https://vkehfdl1.github.io/AutoRAG-Research/datasets/text/open-ragbench/) | arXiv PDF RAG with generation ground truth and multimodal understanding |

## Available Pipelines

SOTA pipelines implemented from papers, ready to run. There are two ways to build a RAG pipeline:

### 1. Retrieval Pipeline

Standalone retrieval pipelines. Use them on their own for retrieval-only evaluation. If you also want to evaluate generation quality, combine any retrieval pipeline with an LLM using the **BasicRAG** generation pipeline — it takes a retrieval pipeline as input, feeds the retrieved results to an LLM, and produces generated answers you can evaluate with generation metrics.

| Pipeline | Description | Reference |
|----------|-------------|-----------|
| [Vector Search](https://vkehfdl1.github.io/AutoRAG-Research/pipelines/retrieval/vector-search/) | Dense vector similarity search (single-vector and multi-vector MaxSim) | - |
| [BM25](https://vkehfdl1.github.io/AutoRAG-Research/pipelines/retrieval/bm25/) | Sparse full-text retrieval | - |
| [HyDE](https://vkehfdl1.github.io/AutoRAG-Research/pipelines/retrieval/hyde/) | Hypothetical Document Embeddings | [Gao et al., 2022](https://arxiv.org/abs/2212.10496) |
| [Hybrid RRF](https://vkehfdl1.github.io/AutoRAG-Research/pipelines/retrieval/hybrid/) | Reciprocal Rank Fusion of two retrieval pipelines | - |
| [Hybrid CC](https://vkehfdl1.github.io/AutoRAG-Research/pipelines/retrieval/hybrid/) | Convex Combination fusion with score normalization | - |

### 2. Generation Pipeline

These pipelines handle retrieval and generation together as a single algorithm. Each implements a specific paper's approach end-to-end.

| Pipeline | Description | Reference |
|----------|-------------|-----------|
| BasicRAG | Any retrieval pipeline + LLM | - |
| IRCoT | Interleaving Retrieval with Chain-of-Thought | [Trivedi et al., ACL 2023](https://arxiv.org/abs/2212.10509) |
| ET2RAG | Majority voting on context subsets | [arXiv:2511.01059](https://arxiv.org/abs/2511.01059) |
| VisRAG-Gen | Vision-language model generation from retrieved images | - |
| MAIN-RAG | Multi-Agent Filtering RAG | [ACL 2025](https://arxiv.org/abs/2501.00332) |

## Available Metrics

**Retrieval** — Set-based: Recall, Precision, F1 / Rank-aware: nDCG, MRR, MAP

**Generation** — N-gram based: BLEU, METEOR, ROUGE / Embedding based: BERTScore, SemScore

> **Missing something?** [Open an issue](https://github.com/vkehfdl1/AutoRAG-Research/issues) and we will implement it. Or check our [Build Your Own](https://vkehfdl1.github.io/AutoRAG-Research/tutorial/) guide.

## Setup

### Install

> We strongly recommend using [uv](https://docs.astral.sh/uv/) as your virtual environment manager. If you use uv, you **must** activate the virtual environment first — otherwise the CLI will not use your uv environment.

**Option 1: Install Script (Recommended)**

The install script handles Python environment, package installation, and PostgreSQL setup in one go.

```bash
curl -LsSf https://raw.githubusercontent.com/NomaDamas/AutoRAG-Research/main/scripts/install.sh -o install.sh
bash install.sh
```

<details>
<summary><b>Manual Install</b></summary>

1. Create and activate a virtual environment (Python 3.10+):

```bash
# uv (recommended)
uv venv .venv --python ">=3.10"
source .venv/bin/activate

# or standard venv
python3 -m venv .venv
source .venv/bin/activate
```

2. Install the package:

```bash
# uv (recommended)
uv pip install autorag-research

# or pip
pip install autorag-research
```

3. Set up PostgreSQL with VectorChord (Docker recommended):

```bash
autorag-research init
cd postgresql && docker compose up -d
```

4. Initialize configuration files:

```bash
autorag-research init
```

This creates `configs/` with database, pipeline, metric, and experiment YAML files.

</details>

### Quick Start

```bash
# 1. See available datasets
autorag-research show datasets

# 2. Ingest a dataset
autorag-research ingest --name beir --extra dataset-name=scifact

# 3. Edit experiment config — choose pipelines and metrics
vim configs/experiment.yaml

# 4. Run your experiment
autorag-research run --db-name=beir_scifact_test

# 5. View results
autorag-research show databases   # see all available result databases
```

`configs/experiment.yaml` is where you define which pipelines and metrics to run:

```yaml
db_name: beir_scifact_test

pipelines:
  retrieval: [bm25, vector_search]
  generation: [basic_rag]

metrics:
  retrieval: [recall, ndcg]
  generation: [rouge]
```

For the full YAML configuration guide, see the [Documentation](https://nomadamas.github.io/AutoRAG-Research/cli/).

### Commands

| Command | Description |
|---------|-------------|
| `autorag-research init` | Download default config files to `./configs/` |
| `autorag-research show datasets` | List available pre-built datasets to download |
| `autorag-research show ingestors` | List available data ingestors and their parameters |
| `autorag-research show pipelines` | List available pipeline configurations |
| `autorag-research show metrics` | List available evaluation metrics |
| `autorag-research show databases` | List ingested database schemas |
| `autorag-research ingest --name <name>` | Ingest a dataset into PostgreSQL |
| `autorag-research run --db-name <name>` | Run experiment with configured pipelines and metrics |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `POSTGRES_PASSWORD` | PostgreSQL password (recommended for security) |
| `AUTORAG_CONFIG_PATH` | Default configuration directory path |


## Implementing New Pipelines (with Claude Code)

This project includes specialized Claude Code agents for implementing new RAG pipelines from research papers.

### Quick Start

```bash
# Full workflow from paper to validated code
/implement-pipeline https://arxiv.org/abs/2212.10496
```
