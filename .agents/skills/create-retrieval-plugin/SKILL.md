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

Create a custom retrieval pipeline plugin for AutoRAG-Research. Retrieval pipelines search a corpus of chunks and return the most relevant results for a given query.

## Architecture Overview

A retrieval plugin consists of two classes:

- **Config class** — extends `BaseRetrievalPipelineConfig`, defines parameters and wires up the pipeline class
- **Pipeline class** — extends `BaseRetrievalPipeline`, implements the actual retrieval logic

The pipeline interacts with the database via `RetrievalPipelineService` (initialized by the base class). Results are stored as `ChunkRetrievedResult` rows.

### Class Hierarchy

```
BasePipelineConfig (config.py)
  └── BaseRetrievalPipelineConfig (config.py)
        └── YourPipelineConfig

BasePipeline (pipelines/base.py)
  └── BaseRetrievalPipeline (pipelines/retrieval/base.py)
        └── YourPipeline
```

### Required Methods

**Config class** (`BaseRetrievalPipelineConfig` subclass):
- `get_pipeline_class()` — return the pipeline class type
- `get_pipeline_kwargs()` — return kwargs passed to the pipeline constructor

**Pipeline class** (`BaseRetrievalPipeline` subclass):
- `_retrieve_by_id(query_id, top_k)` — async retrieval using a query ID (query exists in DB with stored embedding)
- `_retrieve_by_text(query_text, top_k)` — async retrieval using raw query text (may trigger embedding computation)
- `_get_pipeline_config()` — return a config dict for DB storage

## Steps

### Step 1: Scaffold the plugin

```bash
autorag-research plugin create my_search --type=retrieval
```

This creates a `my_search_plugin/` directory with:
```
my_search_plugin/
├── pyproject.toml          # Entry point: autorag_research.pipelines
├── src/
│   └── my_search_plugin/
│       ├── __init__.py
│       ├── pipeline.py     # Config + Pipeline skeleton
│       └── retrieval/
│           └── my_search.yaml   # YAML config
└── tests/
    └── test_my_search.py
```

### Step 2: Understand the generated skeleton

Open `src/my_search_plugin/pipeline.py`. You'll see two classes:

**Config class:**
```python
@dataclass(kw_only=True)
class MySearchPipelineConfig(BaseRetrievalPipelineConfig):
    pipeline_type: PipelineType = field(default=PipelineType.RETRIEVAL, init=False)

    def get_pipeline_class(self) -> type["MySearchPipeline"]:
        return MySearchPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        return {}
```

**Pipeline class:**
```python
class MySearchPipeline(BaseRetrievalPipeline):
    def __init__(self, session_factory, name, schema=None, **kwargs):
        super().__init__(session_factory, name, schema)

    def _get_pipeline_config(self) -> dict[str, Any]:
        return {"type": "my_search"}

    async def _retrieve_by_id(self, query_id, top_k) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def _retrieve_by_text(self, query_text, top_k) -> list[dict[str, Any]]:
        raise NotImplementedError
```

### Step 3: Implement the retrieval logic

> **DO NOT add your own `asyncio.gather`, `asyncio.Semaphore`, or any concurrency control inside
> `_retrieve_by_id` or `_retrieve_by_text`.** The base pipeline's `run()` method already handles
> parallel execution of all queries via `run_with_concurrency_limit()` (semaphore + gather),
> controlled by the `max_concurrency` config parameter. Your method is called once per query —
> just implement the retrieval logic for that single query and return the results.

Each retrieval method must return a list of dicts with `doc_id` (chunk ID) and `score`:

```python
async def _retrieve_by_id(self, query_id: int | str, top_k: int) -> list[dict[str, Any]]:
    # Access the database service (initialized by base class)
    # self._service: RetrievalPipelineService

    # Example: get query embedding from DB, then search
    query = self._service.find_query_by_id(query_id)
    # ... your custom retrieval logic for THIS SINGLE QUERY ...
    return [{"doc_id": chunk_id, "score": relevance_score}, ...]

async def _retrieve_by_text(self, query_text: str, top_k: int) -> list[dict[str, Any]]:
    # Used for ad-hoc queries not in the DB
    # May need to compute embeddings on-the-fly
    return [{"doc_id": chunk_id, "score": relevance_score}, ...]
```

**Adding custom parameters:**

Add fields to your config class and pass them through `get_pipeline_kwargs()`:

```python
@dataclass(kw_only=True)
class MySearchPipelineConfig(BaseRetrievalPipelineConfig):
    similarity_threshold: float = 0.5
    index_name: str = "my_index"
    pipeline_type: PipelineType = field(default=PipelineType.RETRIEVAL, init=False)

    def get_pipeline_class(self) -> type["MySearchPipeline"]:
        return MySearchPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        return {
            "similarity_threshold": self.similarity_threshold,
            "index_name": self.index_name,
        }
```

Then accept them in the pipeline constructor:

```python
class MySearchPipeline(BaseRetrievalPipeline):
    def __init__(self, session_factory, name, schema=None,
                 similarity_threshold=0.5, index_name="my_index", **kwargs):
        super().__init__(session_factory, name, schema)
        self.similarity_threshold = similarity_threshold
        self.index_name = index_name
```

### Step 4: Update the YAML config

Edit `src/my_search_plugin/retrieval/my_search.yaml`:

```yaml
_target_: my_search_plugin.pipeline.MySearchPipelineConfig
description: "My custom search retrieval"
name: my_search
similarity_threshold: 0.5
index_name: my_index
top_k: 10
batch_size: 128
max_concurrency: 16
max_retries: 3
retry_delay: 1.0
```

The `_target_` field must be the fully qualified import path to your config class.

### Step 5: Write tests

Edit `tests/test_my_search.py`:

```python
from unittest.mock import AsyncMock, MagicMock
from my_search_plugin.pipeline import MySearchPipelineConfig, MySearchPipeline


def test_config():
    config = MySearchPipelineConfig(name="my_search")
    assert config.get_pipeline_class() is MySearchPipeline
    assert config.name == "my_search"


def test_config_custom_params():
    config = MySearchPipelineConfig(
        name="my_search",
        similarity_threshold=0.8,
    )
    kwargs = config.get_pipeline_kwargs()
    assert kwargs["similarity_threshold"] == 0.8
```

For integration tests that need a database, use the `db_session` fixture from AutoRAG-Research's test infrastructure.

### Step 6: Install and register

```bash
cd my_search_plugin
pip install -e .
# or with uv:
# uv pip install -e .

# Sync YAML configs into the project
cd ..
autorag-research plugin sync
```

### Step 7: Verify

Check that the config was copied:
```bash
ls configs/pipelines/retrieval/my_search.yaml
```

Use it in your experiment config (`configs/experiment.yaml`) by referencing the pipeline name.

## Reference

### Key Files
- `autorag_research/config.py` — `BasePipelineConfig`, `BaseRetrievalPipelineConfig`
- `autorag_research/pipelines/base.py` — `BasePipeline`
- `autorag_research/pipelines/retrieval/base.py` — `BaseRetrievalPipeline`
- `autorag_research/orm/service/retrieval_pipeline.py` — `RetrievalPipelineService`
- `autorag_research/plugin_registry.py` — Entry point discovery

### Example Implementations
- `autorag_research/pipelines/retrieval/bm25.py` — BM25 retrieval
- `autorag_research/pipelines/retrieval/vector_search.py` — Vector similarity search
- `autorag_research/pipelines/retrieval/hybrid.py` — Hybrid (BM25 + vector) retrieval
- `autorag_research/pipelines/retrieval/hyde.py` — HyDE (Hypothetical Document Embeddings)

### YAML Config Examples
- `configs/pipelines/retrieval/bm25.yaml`
- `configs/pipelines/retrieval/vector_search.yaml`
- `configs/pipelines/retrieval/hybrid_rrf.yaml`
