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

Create a custom data ingestor for AutoRAG-Research. Ingestors load external datasets (e.g., from HuggingFace, local files, APIs) into the AutoRAG-Research database. Unlike pipeline/metric plugins, ingestors use a decorator-based registration pattern.

## Architecture Overview

An ingestor plugin consists of:

- **Ingestor class** — extends `TextEmbeddingDataIngestor` (for text) or `MultiModalEmbeddingDataIngestor` (for images), decorated with `@register_ingestor`
- **No config class needed** — CLI parameters are auto-extracted from `__init__` type hints

### Class Hierarchy

```
DataIngestor (data/base.py)
  ├── TextEmbeddingDataIngestor (data/base.py)
  │     └── YourIngestor
  └── MultiModalEmbeddingDataIngestor (data/base.py)
        └── YourMultiModalIngestor
```

### Required Methods

- `__init__(embedding_model, ...)` — accept embedding model + dataset-specific params
- `detect_primary_key_type()` — return `"bigint"` or `"string"` based on dataset IDs
- `ingest(subset, query_limit, min_corpus_cnt)` — load data and save to database via `self.service`

### How Registration Works

The `@register_ingestor` decorator:
1. Registers the class in a global registry at import time
2. Auto-extracts CLI parameters from `__init__` type hints (`Literal` -> choices, defaults -> optional)
3. Makes the ingestor available via `autorag-research ingest <name>`

External plugins are discovered via the `autorag_research.ingestors` entry point group.

## Steps

### Step 1: Create the plugin project

Unlike pipeline/metric plugins, ingestors don't have a scaffold command. Create the project manually:

```
my_dataset_plugin/
├── pyproject.toml
├── src/
│   └── my_dataset_plugin/
│       ├── __init__.py
│       └── ingestor.py
└── tests/
    └── test_ingestor.py
```

### Step 2: Write pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my-dataset-plugin"
version = "0.1.0"
description = "AutoRAG-Research ingestor for My Dataset"
requires-python = ">=3.10"
dependencies = [
    "autorag-research",
]

[project.entry-points."autorag_research.ingestors"]
my_dataset = "my_dataset_plugin.ingestor"

[tool.hatch.build.targets.wheel]
packages = ["src/my_dataset_plugin"]
```

The entry point group must be `autorag_research.ingestors`. When the module is imported, `@register_ingestor` triggers registration.

### Step 3: Implement the ingestor

Create `src/my_dataset_plugin/ingestor.py`:

```python
import logging
from typing import Literal

from langchain_core.embeddings import Embeddings

from autorag_research.data.base import TextEmbeddingDataIngestor
from autorag_research.data.registry import register_ingestor
from autorag_research.exceptions import ServiceNotSetError

logger = logging.getLogger("AutoRAG-Research")

MY_DATASETS = Literal["subset_a", "subset_b", "subset_c"]


@register_ingestor(
    name="my_dataset",
    description="My custom dataset for RAG evaluation",
    hf_repo="my-dataset-dumps",  # Optional: HuggingFace repo suffix for pre-ingested data
)
class MyDatasetIngestor(TextEmbeddingDataIngestor):
    def __init__(
        self,
        embedding_model: Embeddings,
        dataset_name: MY_DATASETS,
    ):
        super().__init__(embedding_model)
        self.dataset_name = dataset_name
        # Download or prepare your dataset here
        # e.g., from HuggingFace: datasets.load_dataset("my_org/my_dataset", dataset_name)

    def detect_primary_key_type(self) -> Literal["bigint", "string"]:
        """Check if dataset uses integer or string IDs."""
        # Inspect a sample of your data
        # Return "bigint" for numeric IDs, "string" for string IDs
        return "string"

    def ingest(
        self,
        subset: Literal["train", "dev", "test"] = "test",
        query_limit: int | None = None,
        min_corpus_cnt: int | None = None,
    ) -> None:
        """Load dataset and save to database."""
        if self.service is None:
            raise ServiceNotSetError

        # 1. Load your dataset
        # corpus: dict[str, dict] = {...}  # id -> {"title": ..., "text": ...}
        # queries: dict[str, str] = {...}  # id -> query text
        # qrels: dict[str, dict[str, int]] = {...}  # query_id -> {doc_id: relevance}

        # 2. Optionally limit queries
        # if query_limit:
        #     queries = dict(list(queries.items())[:query_limit])

        # 3. Save documents/pages/chunks
        # self.service.save_documents_and_chunks(corpus, ...)

        # 4. Save queries with ground truth
        # self.service.save_queries_with_gt(queries, qrels, ...)

        logger.info(f"Ingested {self.dataset_name} ({subset})")
```

### Key Implementation Details

**`__init__` parameter types drive CLI generation:**

| Type Hint | CLI Behavior |
|---|---|
| `Literal["a", "b"]` | `--param` with choices `[a, b]`, required |
| `str` | `--param`, required string |
| `int = 100` | `--param`, optional with default 100 |
| `bool = False` | `--param/--no-param` flag |
| `list[str]` | `--param` accepting comma-separated values |

Parameters named `embedding_model` or `late_interaction_embedding_model` are automatically skipped (injected by the CLI).

**Using `self.service` (TextDataIngestionService):**

The service is injected by the CLI after construction via `set_service()`. Key methods:

- `self.service.save_document(doc_id, title)` — save a document
- `self.service.save_page(page_id, doc_id, content)` — save a page
- `self.service.save_chunk(chunk_id, page_id, content)` — save a chunk
- `self.service.save_query(query_id, text, ground_truth_ids)` — save a query with GT

Refer to existing ingestors for the exact service method signatures.

### Step 4: Write tests

Create `tests/test_ingestor.py`:

```python
from unittest.mock import MagicMock

from my_dataset_plugin.ingestor import MyDatasetIngestor


def test_ingestor_init():
    embedding_model = MagicMock()
    ingestor = MyDatasetIngestor(
        embedding_model=embedding_model,
        dataset_name="subset_a",
    )
    assert ingestor.dataset_name == "subset_a"


def test_detect_primary_key_type():
    embedding_model = MagicMock()
    ingestor = MyDatasetIngestor(
        embedding_model=embedding_model,
        dataset_name="subset_a",
    )
    result = ingestor.detect_primary_key_type()
    assert result in ("bigint", "string")
```

### Step 5: Install and register

```bash
cd my_dataset_plugin
pip install -e .
# or: uv add --editable ./my_dataset_plugin
```

No `plugin sync` needed for ingestors — they are discovered automatically via entry points.

### Step 6: Verify

```bash
# The ingestor should appear in the CLI
autorag-research ingest --help

# Use it
autorag-research ingest my_dataset --dataset-name subset_a
```

## Reference

### Key Files
- `autorag_research/data/base.py` — `DataIngestor`, `TextEmbeddingDataIngestor`, `MultiModalEmbeddingDataIngestor`
- `autorag_research/data/registry.py` — `@register_ingestor` decorator, `IngestorMeta`, `ParamMeta`
- `autorag_research/orm/service/text_ingestion.py` — `TextDataIngestionService`
- `autorag_research/orm/service/multi_modal_ingestion.py` — `MultiModalIngestionService`

### Example Implementations
- `autorag_research/data/beir.py` — BEIR benchmark datasets
- `autorag_research/data/bright.py` — BRIGHT dataset
- `autorag_research/data/mrtydi.py` — Mr. TyDi multilingual dataset
- `autorag_research/data/ragbench.py` — RAGBench dataset

### Parameters Skipped by Registry
The following `__init__` parameters are automatically excluded from CLI generation (they are injected by the framework):
- `self`
- `embedding_model`
- `late_interaction_embedding_model`
