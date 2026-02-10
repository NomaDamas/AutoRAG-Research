# Generation Pipeline Plugin

Implement a custom generation pipeline and distribute it as a plugin.

## Overview

| Field | Value |
|-------|-------|
| Entry Point Group | `autorag_research.pipelines` |
| Config Base Class | `BaseGenerationPipelineConfig` |
| Pipeline Base Class | `BaseGenerationPipeline` |
| Config Module | `autorag_research.config` |
| Pipeline Module | `autorag_research.pipelines.generation.base` |

A generation pipeline composes a retrieval pipeline with an LLM to produce
answers. The config declares model settings and which retrieval pipeline to use;
the pipeline class implements the retrieve-then-generate logic.

## Scaffold

Generate boilerplate with the CLI:

```bash
autorag-research plugin create my_rag --type=generation
```

This creates a package under `my_rag_plugin/` with config, pipeline, YAML, and
`pyproject.toml` pre-wired.

## Config Class

Subclass `BaseGenerationPipelineConfig` and define custom parameters. Implement
`get_pipeline_class()` and `get_pipeline_kwargs()`.

```python
from dataclasses import dataclass, field
from typing import Any

from autorag_research.config import BaseGenerationPipelineConfig, PipelineType


@dataclass(kw_only=True)
class MyRAGPipelineConfig(BaseGenerationPipelineConfig):
    """Configuration for MyRAG generation pipeline."""

    pipeline_type: PipelineType = field(default=PipelineType.GENERATION, init=False)

    # Add custom config fields
    temperature: float = 0.7
    system_prompt: str = "You are a helpful assistant."

    def get_pipeline_class(self) -> type["MyRAGPipeline"]:
        return MyRAGPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "system_prompt": self.system_prompt,
        }
```

### Inherited Fields

In addition to the base fields shared with retrieval configs (`name`,
`description`, `top_k`, `batch_size`, `max_concurrency`, `max_retries`,
`retry_delay`), generation configs add:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `llm` | `str \| BaseLanguageModel` | required | LLM model name (auto-loaded) or LangChain instance |
| `retrieval_pipeline_name` | `str` | required | Name of retrieval pipeline to compose with |

When `llm` is a string such as `"gpt-4o-mini"`, the framework loads it
automatically via `load_llm()`. The `retrieval_pipeline_name` references a
retrieval pipeline already registered in the experiment; the Executor resolves
and injects it at runtime.

### Abstract Methods

| Method | Returns | Purpose |
|--------|---------|---------|
| `get_pipeline_class()` | `type[BaseGenerationPipeline]` | Pipeline class to instantiate |
| `get_pipeline_kwargs()` | `dict[str, Any]` | Custom kwargs passed to the pipeline constructor (beyond `session_factory`, `name`, `llm`, `retrieval_pipeline`, `schema` which are injected automatically) |

## Pipeline Class

Subclass `BaseGenerationPipeline` and implement `_generate()` and
`_get_pipeline_config()`.

```python
from typing import Any

from langchain_core.language_models import BaseLanguageModel
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.orm.service.generation_pipeline import GenerationResult
from autorag_research.pipelines.generation.base import BaseGenerationPipeline
from autorag_research.pipelines.retrieval.base import BaseRetrievalPipeline


class MyRAGPipeline(BaseGenerationPipeline):
    """MyRAG generation pipeline."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        llm: BaseLanguageModel,
        retrieval_pipeline: BaseRetrievalPipeline,
        schema: Any | None = None,
        temperature: float = 0.7,
        system_prompt: str = "You are a helpful assistant.",
    ):
        super().__init__(session_factory, name, llm, retrieval_pipeline, schema)
        self.temperature = temperature
        self.system_prompt = system_prompt

    def _get_pipeline_config(self) -> dict[str, Any]:
        return {
            "type": "my_rag",
            "temperature": self.temperature,
        }

    async def _generate(self, query_id: int, top_k: int) -> GenerationResult:
        # Step 1: Retrieve relevant chunks
        results = await self._retrieval_pipeline._retrieve_by_id(query_id, top_k)
        chunk_ids = [r["doc_id"] for r in results]
        chunk_contents = self._service.get_chunk_contents(chunk_ids)

        # Step 2: Get query text
        query_text = self._get_query_text(query_id)

        # Step 3: Build prompt and generate
        context = "\n\n".join(chunk_contents)
        prompt = f"{self.system_prompt}\n\nContext:\n{context}\n\nQuestion: {query_text}\n\nAnswer:"
        response = await self._llm.ainvoke(prompt)

        return GenerationResult(text=str(response.content))
```

### Available Resources

Inside `_generate()`, you have access to:

| Resource | Access | Description |
|----------|--------|-------------|
| Retrieval pipeline | `self._retrieval_pipeline` | Composed retrieval pipeline instance |
| LLM | `self._llm` | LangChain `BaseLanguageModel` (use `.ainvoke()` for async) |
| Service | `self._service` | `GenerationPipelineService` for DB operations |
| Query text | `self._get_query_text(query_id)` | Gets query text (uses `query_to_llm` if available) |

### GenerationResult

`_generate()` must return a `GenerationResult` dataclass:

```python
@dataclass
class GenerationResult:
    text: str                              # Generated answer text
    token_usage: dict[str, int] | None = None  # Optional: {"prompt": N, "completion": M}
    metadata: dict[str, Any] | None = None     # Optional: extra metadata
```

Only `text` is required. Populate `token_usage` if your LLM response includes
token counts -- the executor persists these for cost tracking. Use `metadata` for
any additional information you want stored alongside the result.

## YAML Configuration

Place a YAML file inside your plugin package:

```yaml
# src/my_rag_plugin/generation/my_rag.yaml
_target_: my_rag_plugin.pipeline.MyRAGPipelineConfig
description: "MyRAG generation pipeline"
name: my_rag
retrieval_pipeline_name: bm25
llm: gpt-4o-mini
top_k: 10
temperature: 0.7
system_prompt: "You are a helpful assistant."
```

`_target_` is the fully-qualified class name of your config dataclass (Hydra-style
instantiation). The remaining keys map directly to dataclass fields.

`retrieval_pipeline_name` must match the `name` field of a retrieval pipeline
config in the same experiment. The Executor resolves this reference and injects
the live pipeline instance into your generation pipeline at runtime.

## Entry Points

Register under the same group as retrieval plugins:

```toml
[project.entry-points."autorag_research.pipelines"]
my_rag = "my_rag_plugin"
```

The key (`my_rag`) is the plugin name. The value is the top-level package
containing your YAML configs.

After installing the package, run:

```bash
autorag-research plugin sync
```

This copies your YAML files into the project's `configs/` directory.

## Testing

Test the config independently. Use `MagicMock()` for the `llm` field to avoid
loading a real model:

```python
from unittest.mock import MagicMock

from my_rag_plugin.pipeline import MyRAGPipelineConfig


def test_config():
    config = MyRAGPipelineConfig(
        name="my_rag",
        llm=MagicMock(),
        retrieval_pipeline_name="bm25",
    )
    assert config.name == "my_rag"


def test_config_custom_fields():
    config = MyRAGPipelineConfig(
        name="my_rag",
        llm=MagicMock(),
        retrieval_pipeline_name="bm25",
        temperature=0.3,
        system_prompt="Answer concisely.",
    )
    kwargs = config.get_pipeline_kwargs()
    assert kwargs["temperature"] == 0.3
    assert kwargs["system_prompt"] == "Answer concisely."
```

For integration tests that exercise `_generate()`, use `FakeListLLM` from
`langchain_core.llms` and the `db_session` fixture from the test `conftest.py`.

## Next

- [Retrieval Pipeline](retrieval-pipeline.md) -- build a retrieval pipeline plugin
- [Metrics](metrics.md) -- implement a custom metric plugin
- [Best Practices](best-practices.md) -- packaging, versioning, and distribution tips
