---
name: create-ingestor-plugin
description: |
  Guide developers through creating a custom data ingestor plugin for AutoRAG-Research.
  Ingestors load external datasets (HuggingFace, local files, APIs) into the database.
  Uses @register_ingestor decorator for automatic CLI parameter extraction. Use when
  ingesting a new dataset format into AutoRAG-Research.
allowed-tools:
  - Bash
  - Read
  - Write
  - Edit
---

# Create Ingestor Plugin

## Workflow

### 1. Scaffold

```bash
autorag-research plugin create my_dataset --type=ingestor
```

Read the generated `ingestor.py`, `pyproject.toml`, and test file to understand the structure.

The generated `pyproject.toml` registers the `autorag_research.ingestors` entry point. The `@register_ingestor` decorator handles automatic CLI parameter extraction from `__init__` type hints.

### 2. Implement the ingestor

**Required methods:**
- `__init__(embedding_model, ...)` — accept embedding model + dataset-specific params
- `detect_primary_key_type()` → `"bigint"` or `"string"`
- `ingest(subset, query_limit, min_corpus_cnt)` — load data and save via `self.service`

**`__init__` type hints drive CLI generation automatically:**

| Type Hint | CLI Behavior |
|---|---|
| `Literal["a", "b"]` | `--param` with choices, required |
| `str` | `--param`, required |
| `int = 100` | `--param`, optional with default |
| `bool = False` | `--param/--no-param` flag |

Parameters named `embedding_model` or `late_interaction_embedding_model` are auto-skipped (injected by CLI).

**`self.service`** is injected after construction via `set_service()`. Read existing ingestors for exact service method signatures.

### 3. Install and verify

```bash
cd my_dataset_plugin
pip install -e .   # or: uv pip install -e .
```

No `plugin sync` needed — ingestors are discovered automatically via entry points.

```bash
autorag-research ingest my_dataset --dataset-name subset_a
```

## Key Files

| Purpose | Path |
|---|---|
| Base classes | `autorag_research/data/base.py` → `TextEmbeddingDataIngestor`, `MultiModalEmbeddingDataIngestor` |
| Registration decorator | `autorag_research/data/registry.py` → `@register_ingestor` |
| Text ingestion service | `autorag_research/orm/service/text_ingestion.py` |
| Multi-modal ingestion service | `autorag_research/orm/service/multi_modal_ingestion.py` |

## Examples

Study these existing implementations for patterns:

- `autorag_research/data/beir.py` — BEIR benchmark (simple, good starting point)
- `autorag_research/data/bright.py` — BRIGHT dataset
- `autorag_research/data/mrtydi.py` — Mr. TyDi multilingual dataset
- `autorag_research/data/ragbench.py` — RAGBench dataset
