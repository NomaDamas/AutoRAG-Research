![Thumbnail](./assets/thumbnail.png)

Automate your RAG research

 [Documentation](https://nomadamas.github.io/AutoRAG-Research/)


## What is AutoRAG-Research?

| Problem | What AutoRAG-Research does                                                       |
|---------|----------------------------------------------------------------------------------|
| Every dataset has a different format. | We unify the formats and pre-computed embeddings for you. Just download and use. |
| Comparing against SOTA pipelines requires implementing each one. | We implement SOTA pipelines from papers. Benchmark yours against them.           |
| Every paper claims SOTA. Which one actually is? | Run all pipelines on your data with one command and compare.                     |

Which pipeline is really SOTA? What datasets are out there? Find it all here.


## Available Datasets

We provide pre-processed datasets with unified formats. Some include **pre-computed embeddings**.

**Text**

| Dataset | Pipeline Support | Description |
|---------|:----------------:|-------------|
| [BEIR](https://arxiv.org/pdf/2104.08663) | Retrieval | Standard IR benchmark across 14 diverse domains (scifact, nq, hotpotqa, ...) |
| [MTEB](https://aclanthology.org/2023.eacl-main.148.pdf) | Retrieval | Large-scale embedding benchmark with any MTEB retrieval task |
| [RAGBench](https://arxiv.org/pdf/2407.11005v1) | Retrieval + Generation | End-to-end RAG evaluation with generation ground truth across 12 domains |
| [MrTyDi](https://aclanthology.org/2021.mrl-1.12.pdf) | Retrieval | Multilingual retrieval across 11 languages |
| [BRIGHT](https://arxiv.org/pdf/2407.12883) | Retrieval + Generation | Reasoning-intensive retrieval with gold answers |

**Image**

| Dataset |    Pipeline Support     | Description |
|---------|:-----------------------:|-------------|
| [ViDoRe](https://arxiv.org/pdf/2407.01449) | Retrieval + Generation* | Visual document QA with 1:1 query-to-page mapping |
| [ViDoRe v2](https://arxiv.org/pdf/2505.17166) |        Retrieval        | Visual document retrieval with corpus-level search |
| [ViDoRe v3](https://arxiv.org/pdf/2601.08620) |        Retrieval        | Visual document retrieval across 8 industry domains |
| [VisRAG](https://arxiv.org/pdf/2410.10594) | Retrieval + Generation* | Vision-based RAG benchmark (ChartQA, SlideVQA, DocVQA, ...) |

**Text + Image**

| Dataset | Pipeline Support | Description |
|---------|:----------------:|-------------|
| [Open-RAGBench](https://huggingface.co/datasets/vectara/open_ragbench) | Retrieval + Generation | arXiv PDF RAG with generation ground truth and multimodal understanding |

> *\* Generation ground truth is available only for some sub-datasets.*

## Available Pipelines

SOTA pipelines implemented from papers, ready to run. There are two ways to build a RAG pipeline:

### 1. Retrieval Pipeline

Standalone retrieval pipelines. Use them on their own for retrieval-only evaluation. If you also want to evaluate generation quality, combine any retrieval pipeline with an LLM using the **BasicRAG** generation pipeline — it takes a retrieval pipeline as input, feeds the retrieved results to an LLM, and produces generated answers you can evaluate with generation metrics.

| Pipeline                                                                         | Description                                                            | Reference |
|----------------------------------------------------------------------------------|------------------------------------------------------------------------|-----------|
| [Vector Search (DPR)](https://aclanthology.org/2020.emnlp-main.550.pdf)          | Dense vector similarity search (single-vector and multi-vector MaxSim) | EMNLP 20  |
| [BM25](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) | Sparse full-text retrieval                                             | -         |
| [HyDE](https://arxiv.org/abs/2212.10496)                                         | Hypothetical Document Embeddings                                       | ACL 23    |
| [Hybrid RRF](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)                | Reciprocal Rank Fusion of two retrieval pipelines                      | -         |
| [Hybrid CC](https://arxiv.org/pdf/2210.11934)                                    | Convex Combination fusion of two retrieval pipelines                   | -         |

### 2. Generation Pipeline

These pipelines handle retrieval and generation together as a single algorithm. Each implements a specific paper's approach end-to-end.

| Pipeline                                     | Description                                            | Reference     |
|----------------------------------------------|--------------------------------------------------------|---------------|
| [BasicRAG](https://arxiv.org/pdf/2005.11401) | Any retrieval pipeline + LLM                           | NeurIPS 20    |
| [IRCoT](https://arxiv.org/abs/2212.10509)    | Interleaving Retrieval with Chain-of-Thought           | ACL 23        |
| [ET2RAG](https://arxiv.org/abs/2511.01059)   | Majority voting on context subsets                     | Preprint / 25 |
| [VisRAG](https://arxiv.org/abs/2410.10594)   | Vision-language model generation from retrieved images | ICLR 25       |
| [MAIN-RAG](https://arxiv.org/abs/2501.00332) | Multi-Agent Filtering RAG                              | ACL 25        |

## Available Metrics

**Retrieval** — Set-based: Recall, Precision, F1 / Rank-aware: nDCG, MRR, MAP

**Generation** — N-gram based: BLEU, METEOR, ROUGE / Embedding based: BERTScore, SemScore

> **Missing something?** [Open an issue](https://github.com/vkehfdl1/AutoRAG-Research/issues) and we will implement it. Or check our [Plugin](https://nomadamas.github.io/AutoRAG-Research/plugins/) guide.

## Setup

### Install

> We strongly recommend using [uv](https://docs.astral.sh/uv/) as your virtual environment manager. If you use uv, you **must** activate the virtual environment first — otherwise the CLI will not use your uv environment.

**Option 1: Install Script (Recommended, Mac OS / Linux)**

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
uv add autorag-research

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
Now you can edit YAML files to setup your own experiments.

</details>

### Quick Start

```bash
# 1. See available datasets
autorag-research show datasets

# 2-1. Ingest a dataset
autorag-research ingest --name beir --extra dataset-name=scifact

# 2-2. Or download a pre-ingested dataset including pre-computed embeddings
autorag-rsearch show datasets beir # type your ingestor name to see if pre-ingested versions are available
autorag-research data restore beir beir_arguana_test_qwen_3_0.6b # example command

# 3. Configure LLM — pick or create a config in configs/llm/
vim configs/llm/openai-gpt5-mini.yaml
# You should set your embedding models in embedding/ folder if needed

# 4. Edit experiment config — choose pipelines and metrics
vim configs/experiment.yaml

# 5. Check your DB connection
vim configs/db.yaml

# 6. Run your experiment
autorag-research run --db-name=beir_scifact_test

# 7. View results in a Gradio leaderboard UI (need to load your env variable for DB connection)
python -m autorag_research.reporting.ui
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

Generation pipelines (and some retrieval pipelines like HyDE) require an LLM. The `llm` field in each pipeline config references a file in `configs/llm/` by name (without `.yaml`):

```yaml
# configs/pipelines/generation/basic_rag.yaml
llm: openai-gpt5-mini   # → loads configs/llm/openai-gpt5-mini.yaml
```

Pre-configured LLM options include `anthropic-claude-4.5-sonnet`, `openai-gpt5-mini`, `google-gemini-3-flash`, `ollama`, `vllm`, and more. See all options in `configs/llm/`.

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

You can also type `--help` in any command to see detailed usage instructions.
Also, we provide a [CLI Reference](https://nomadamas.github.io/AutoRAG-Research/cli/).


## Build Your Own Plugin

AutoRAG-Research supports a plugin system so you can add your own retrieval pipelines, generation pipelines, or evaluation metrics — and use them alongside the built-in ones in the same experiment.

A plugin is a standalone Python package. You implement your logic, register it via Python's `entry_points`, and the framework discovers and loads it automatically. No need to fork the repo or modify the core codebase.

**What you can build:**

| Plugin Type | What it does | Base Class |
|-------------|--------------|------------|
| Retrieval Pipeline | Custom search/retrieval logic | `BaseRetrievalPipeline` |
| Generation Pipeline | Custom retrieve-then-generate logic | `BaseGenerationPipeline` |
| Retrieval Metric | Custom retrieval evaluation metric | `BaseRetrievalMetricConfig` |
| Generation Metric | Custom generation evaluation metric | `BaseGenerationMetricConfig` |

**How it works:**

```bash
# 1. Scaffold — generates a ready-to-edit project with config, code, YAML, and tests
autorag-research plugin create my_search --type=retrieval

# 2. Implement — edit the generated pipeline.py (or metric.py)
cd my_search_plugin
vim src/my_search_plugin/pipeline.py

# 3. Install — register the plugin in your environment
pip install -e .

# 4. Sync — copy the plugin's YAML config into your project's configs/ directory
autorag-research plugin sync

# 5. Use — add it to experiment.yaml and run like any built-in pipeline
autorag-research run --db-name=my_dataset
```

After `plugin sync`, your plugin appears in `configs/pipelines/` or `configs/metrics/` and can be referenced in `experiment.yaml` just like any built-in component.

For the full implementation guide, see the [Plugin Documentation](https://nomadamas.github.io/AutoRAG-Research/plugins/).


## Contributing

We are open source project and always welcome contributions who love RAG! Feel free to open issues or submit pull requests on GitHub.
You can check our [Contribution Guide](https://nomadamas.github.io/AutoRAG-Research/contributing/) for more details.


## Acknowledgements

This project is made by the creator of [AutoRAG](https://github.com/Marker-Inc-Korea/AutoRAG), Jeffrey & Bobb Kim.
All works are done in [NomaDamas](https://github.com/NomaDamas), AI Hacker House in Seoul, Korea.
