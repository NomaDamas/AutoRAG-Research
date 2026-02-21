---
name: create-generation-plugin
description: |
  Guide developers through creating a custom generation pipeline plugin for AutoRAG-Research.
  Walks through scaffolding, implementing BaseGenerationPipeline methods, composing with
  retrieval pipelines, writing YAML configs, testing, and installing. Use when building a
  new RAG generation strategy (e.g., chain-of-thought RAG, multi-hop RAG).
allowed-tools:
  - Bash
  - Read
  - Write
  - Edit
---

# Create Generation Plugin

## Workflow

### 1. Scaffold

```bash
autorag-research plugin create my_rag --type=generation
```

Read the generated `pipeline.py`, `pyproject.toml`, YAML config, and test file to understand the structure.

### 2. Implement

Implement the `_generate(query_id, top_k)` method. This is where your RAG strategy lives.

**Available attributes inside the pipeline:**
- `self._llm` — LangChain `BaseLanguageModel` (use `await self._llm.ainvoke(prompt)`)
- `self._retrieval_pipeline` — composed retrieval pipeline (use `await self._retrieval_pipeline._retrieve_by_id(query_id, top_k)`)
- `self._service` — `GenerationPipelineService` (use `self._service.get_chunk_contents(chunk_ids)`, `self._get_query_text(query_id)`)

Must return a `GenerationResult(text=...)` (from `autorag_research.orm.service.generation_pipeline`).

> **DO NOT add your own `asyncio.gather`, `asyncio.Semaphore`, or any concurrency control.**
> The base pipeline's `run()` already handles parallel execution of all queries via
> `run_with_concurrency_limit()` (semaphore + gather), controlled by the `max_concurrency`
> config parameter. Your `_generate` method is called once per single query — just implement
> the retrieve-and-generate logic for that one query.

**Custom parameters:** Add fields to your config class and pass them via `get_pipeline_kwargs()` → accept them in the pipeline constructor.

**Inherited config fields** (from `BaseGenerationPipelineConfig`):
- `llm` — LLM model string (auto-converted to LangChain model instance)
- `retrieval_pipeline_name` — name of the retrieval pipeline to compose with (Executor injects it)

### 3. Write tests and install

Use `langchain_core.language_models.FakeListLLM` to mock the LLM in tests.

```bash
cd my_rag_plugin
pip install -e .   # or: uv pip install -e .
cd .. && autorag-research plugin sync
```

Verify: `ls configs/pipelines/generation/my_rag.yaml`

## Key Files

| Purpose | Path |
|---|---|
| Base config class | `autorag_research/config.py` → `BaseGenerationPipelineConfig` |
| Base pipeline class | `autorag_research/pipelines/generation/base.py` → `BaseGenerationPipeline` |
| Service + GenerationResult | `autorag_research/orm/service/generation_pipeline.py` |
| Plugin entry point discovery | `autorag_research/plugin_registry.py` |

## Examples

Study these existing implementations for patterns:

- `autorag_research/pipelines/generation/basic_rag.py` — Simple retrieve-then-generate (start here)
- `autorag_research/pipelines/generation/ircot.py` — Interleaving retrieval with chain-of-thought
- `autorag_research/pipelines/generation/et2rag.py` — Entity-aware RAG
- `autorag_research/pipelines/generation/main_rag.py` — Main RAG pipeline
- YAML configs: `configs/pipelines/generation/basic_rag.yaml`, `configs/pipelines/generation/ircot.yaml`
