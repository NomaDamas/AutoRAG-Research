---
name: create-retrieval-plugin
description: |
  Guide developers through creating a custom retrieval pipeline plugin for AutoRAG-Research.
  Walks through scaffolding, implementing BaseRetrievalPipeline methods, writing YAML configs,
  testing, and installing. Use when building a new search/retrieval strategy (e.g., Elasticsearch,
  ColBERT, custom vector search).
allowed-tools:
  - Bash
  - Read
  - Write
  - Edit
---

# Create Retrieval Plugin

## Workflow

### 1. Scaffold

```bash
autorag-research plugin create my_search --type=retrieval
```

Read the generated `pipeline.py`, `pyproject.toml`, YAML config, and test file to understand the structure.

### 2. Implement

Implement the two abstract methods in the pipeline class:

- `_retrieve_by_id(query_id, top_k)` — retrieve using query ID (query exists in DB with stored embedding)
- `_retrieve_by_text(query_text, top_k)` — retrieve using raw text (may need on-the-fly embedding)

Both must return `list[dict[str, Any]]` with `doc_id` (chunk ID) and `score` keys.

> **DO NOT add your own `asyncio.gather`, `asyncio.Semaphore`, or any concurrency control.**
> The base pipeline's `run()` already handles parallel execution of all queries via
> `run_with_concurrency_limit()` (semaphore + gather), controlled by the `max_concurrency`
> config parameter. Your method is called once per single query — just implement the
> retrieval logic for that one query.

**Custom parameters:** Add fields to your config class and pass them via `get_pipeline_kwargs()` → accept them in the pipeline constructor. See `bm25.py` for a real example.

### 3. Write tests and install

```bash
cd my_search_plugin
pip install -e .   # or: uv pip install -e .
cd .. && autorag-research plugin sync
```

Verify: `ls configs/pipelines/retrieval/my_search.yaml`

## Key Files

| Purpose | Path |
|---|---|
| Base config class | `autorag_research/config.py` → `BaseRetrievalPipelineConfig` |
| Base pipeline class | `autorag_research/pipelines/retrieval/base.py` → `BaseRetrievalPipeline` |
| Service layer | `autorag_research/orm/service/retrieval_pipeline.py` → `RetrievalPipelineService` |
| Plugin entry point discovery | `autorag_research/plugin_registry.py` |

## Examples

Study these existing implementations for patterns:

- `autorag_research/pipelines/retrieval/bm25.py` — BM25 retrieval (simple)
- `autorag_research/pipelines/retrieval/vector_search.py` — Vector similarity search
- `autorag_research/pipelines/retrieval/hybrid.py` — Hybrid (BM25 + vector)
- `autorag_research/pipelines/retrieval/hyde.py` — HyDE (Hypothetical Document Embeddings)
- YAML configs: `configs/pipelines/retrieval/bm25.yaml`, `configs/pipelines/retrieval/vector_search.yaml`
