# Retrieval Pipeline Plugin

Implement a custom retrieval pipeline and distribute it as a plugin.

## Overview

| Field | Value |
|-------|-------|
| Entry Point Group | `autorag_research.pipelines` |
| Config Base Class | `BaseRetrievalPipelineConfig` |
| Pipeline Base Class | `BaseRetrievalPipeline` |
| Config Module | `autorag_research.config` |
| Pipeline Module | `autorag_research.pipelines.retrieval.base` |

A retrieval pipeline plugin consists of two classes: a **config** dataclass that
declares parameters and a **pipeline** class that implements search logic. The
config tells the executor *how* to build the pipeline; the pipeline performs the
actual retrieval.

## Scaffold

Generate boilerplate with the CLI:

```bash
autorag-research plugin create my_search --type=retrieval
```

This creates a ready-to-edit package under `my_search_plugin/` with config,
pipeline, YAML, and `pyproject.toml` pre-wired.

## Config Class

Subclass `BaseRetrievalPipelineConfig` and define your custom parameters as
dataclass fields. Implement `get_pipeline_class()` and `get_pipeline_kwargs()`.

```python
from dataclasses import dataclass, field
from typing import Any

from autorag_research.config import BaseRetrievalPipelineConfig, PipelineType


@dataclass(kw_only=True)
class MySearchPipelineConfig(BaseRetrievalPipelineConfig):
    """Configuration for MySearch retrieval pipeline."""

    pipeline_type: PipelineType = field(default=PipelineType.RETRIEVAL, init=False)

    # Add custom config fields
    index_path: str = "/data/index"
    similarity_threshold: float = 0.5

    def get_pipeline_class(self) -> type["MySearchPipeline"]:
        return MySearchPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        return {
            "index_path": self.index_path,
            "similarity_threshold": self.similarity_threshold,
        }
```

### Inherited Fields

Every retrieval config inherits these fields from `BasePipelineConfig`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | required | Unique pipeline instance name |
| `description` | `str` | `""` | Optional description |
| `top_k` | `int` | `10` | Results per query |
| `batch_size` | `int` | `128` | Queries per DB batch |
| `max_concurrency` | `int` | `16` | Max concurrent async operations |
| `max_retries` | `int` | `3` | Retry attempts for failed queries |
| `retry_delay` | `float` | `1.0` | Base delay for exponential backoff |

### Abstract Methods

| Method | Returns | Purpose |
|--------|---------|---------|
| `get_pipeline_class()` | `type[BaseRetrievalPipeline]` | Pipeline class to instantiate |
| `get_pipeline_kwargs()` | `dict[str, Any]` | Custom kwargs passed to the pipeline constructor (beyond `session_factory`, `name`, `schema` which are injected automatically) |
| `get_run_kwargs()` | `dict[str, Any]` | Already implemented by `BaseRetrievalPipelineConfig` -- returns `top_k`, `batch_size`, `max_concurrency`, `max_retries`, and `retry_delay` |

You must implement `get_pipeline_class()` and `get_pipeline_kwargs()`.
`get_run_kwargs()` is provided by the base class and normally does not need
overriding.

## Pipeline Class

Subclass `BaseRetrievalPipeline` and implement the three abstract methods.

```python
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from autorag_research.pipelines.retrieval.base import BaseRetrievalPipeline


class MySearchPipeline(BaseRetrievalPipeline):
    """MySearch retrieval pipeline."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        schema: Any | None = None,
        index_path: str = "/data/index",
        similarity_threshold: float = 0.5,
    ):
        super().__init__(session_factory, name, schema)
        self.index_path = index_path
        self.similarity_threshold = similarity_threshold

    def _get_pipeline_config(self) -> dict[str, Any]:
        return {
            "type": "my_search",
            "index_path": self.index_path,
            "similarity_threshold": self.similarity_threshold,
        }

    async def _retrieve_by_id(self, query_id: int | str, top_k: int) -> list[dict[str, Any]]:
        """Retrieve using query ID (query exists in database).

        Used for batch processing where queries have pre-computed embeddings.
        """
        # Access query embedding from DB via self._service
        # Perform your search logic
        return [{"doc_id": chunk_id, "score": score}]

    async def _retrieve_by_text(self, query_text: str, top_k: int) -> list[dict[str, Any]]:
        """Retrieve using raw query text (may need to compute embedding).

        Used for ad-hoc retrieval and by generation pipelines.
        """
        # Compute embedding on-the-fly if needed
        # Perform your search logic
        return [{"doc_id": chunk_id, "score": score}]
```

### Abstract Methods

| Method | When Called | Use Case |
|--------|-----------|----------|
| `_retrieve_by_id(query_id, top_k)` | `pipeline.run()` batch processing | Queries exist in DB with stored embeddings |
| `_retrieve_by_text(query_text, top_k)` | `pipeline.retrieve()` single query | Ad-hoc queries, used by generation pipelines |
| `_get_pipeline_config()` | Pipeline initialization | Returns dict stored in DB for reproducibility |

### Return Format

Both `_retrieve_by_id` and `_retrieve_by_text` return a list of dicts. Each dict
contains:

| Key | Type | Description |
|-----|------|-------------|
| `doc_id` | `int \| str` | Chunk ID in the database |
| `score` | `float` | Relevance score (higher is better) |

The base class handles persisting these results to `ChunkRetrievedResult` rows
automatically.

## YAML Configuration

Place a YAML file inside your plugin package. The executor loads it via
Hydra-style instantiation.

```yaml
# src/my_search_plugin/retrieval/my_search.yaml
_target_: my_search_plugin.pipeline.MySearchPipelineConfig
description: "MySearch retrieval pipeline"
name: my_search
top_k: 10
batch_size: 128
max_concurrency: 16
index_path: /data/index
similarity_threshold: 0.5
```

`_target_` is the fully-qualified class name of your config dataclass. The
remaining keys map directly to dataclass fields. When the executor loads this
file, it instantiates `MySearchPipelineConfig` with these values.

## Entry Points

Register your plugin in `pyproject.toml` so the framework discovers it at
runtime:

```toml
[project.entry-points."autorag_research.pipelines"]
my_search = "my_search_plugin"
```

The key (`my_search`) is the plugin name used in `plugin sync`. The value is the
top-level package that contains your YAML configs.

After installing the package, run:

```bash
autorag-research plugin sync
```

This copies your YAML files into the project's `configs/` directory.

## Testing

Test the config independently of the database:

```python
from my_search_plugin.pipeline import MySearchPipelineConfig


def test_config():
    config = MySearchPipelineConfig(name="my_search")
    assert config.name == "my_search"
    assert config.get_pipeline_class() is not None


def test_config_custom_fields():
    config = MySearchPipelineConfig(
        name="my_search",
        index_path="/custom/index",
        similarity_threshold=0.8,
    )
    kwargs = config.get_pipeline_kwargs()
    assert kwargs["index_path"] == "/custom/index"
    assert kwargs["similarity_threshold"] == 0.8
```

For integration tests that exercise `_retrieve_by_id` and `_retrieve_by_text`,
use the `db_session` fixture from the test `conftest.py` and seed test data per
the project testing guidelines.

## Next

- [Generation Pipeline](generation-pipeline.md) -- build a generation pipeline plugin
- [Best Practices](best-practices.md) -- packaging, versioning, and distribution tips
