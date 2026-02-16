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

### 3. Database Schema (critical)

Ingestors must populate the correct entity hierarchy:

```
Document → Page → Chunk (text)
                → ImageChunk (images)
```

- **Document** — top-level container (e.g., a Wikipedia article, a PDF)
- **Page** — subdivision within a document (linked via `document_id`)
- **Chunk** — text passage with embedding vector (linked to Page via `PageChunkRelation`)
- **ImageChunk** — image binary with embedding vector (linked to Page via `PageChunkRelation`)
- **Query** — search query with `generation_gt: list[str] | None` (ground truth answers)

**RetrievalRelation** — links queries to relevant chunks using AND/OR group structure:

```
RetrievalRelation(query_id, chunk_id, group_index, group_order, score)

group_index = AND group number
group_order = OR position within the group

Example: query needs (chunk_A OR chunk_B) AND chunk_C
  → (query, chunk_A, group_index=0, group_order=0)
  → (query, chunk_B, group_index=0, group_order=1)
  → (query, chunk_C, group_index=1, group_order=0)
```

This AND/OR structure is critical for multi-hop queries. See `ai_instructions/db_schema.md` for the full DBML schema.

### 4. Install and verify

```bash
cd my_dataset_plugin
pip install -e .   # or: uv pip install -e .
```

No `plugin sync` needed — ingestors are discovered automatically via entry points.

```bash
autorag-research ingest my_dataset --dataset-name subset_a
```

## Testing

Use `ingestor_test_utils` for integration tests against a real PostgreSQL database:

- `IngestorTestConfig` — declare expected counts (queries, chunks, image_chunks), relation checks, primary key type
- `create_test_database(config)` — context manager that creates/drops an isolated test DB
- `IngestorTestVerifier` — runs all configured checks: count verification, format validation, retrieval relation checks, generation_gt checks, content hash verification

See `tests/autorag_research/data/ingestor_test_utils.py` for full API and usage examples in the module docstring.

## Key Files

| Purpose | Path |
|---|---|
| Base classes | `autorag_research/data/base.py` → `TextEmbeddingDataIngestor`, `MultiModalEmbeddingDataIngestor` |
| Registration decorator | `autorag_research/data/registry.py` → `@register_ingestor` |
| Text ingestion service | `autorag_research/orm/service/text_ingestion.py` |
| Multi-modal ingestion service | `autorag_research/orm/service/multi_modal_ingestion.py` |
| DB schema reference | `ai_instructions/db_schema.md` |
| Test utilities | `tests/autorag_research/data/ingestor_test_utils.py` |

## Examples

Study these existing implementations for patterns:

- `autorag_research/data/beir.py` — BEIR benchmark (simple, good starting point)
- `autorag_research/data/bright.py` — BRIGHT dataset
- `autorag_research/data/mrtydi.py` — Mr. TyDi multilingual dataset
- `autorag_research/data/ragbench.py` — RAGBench dataset
