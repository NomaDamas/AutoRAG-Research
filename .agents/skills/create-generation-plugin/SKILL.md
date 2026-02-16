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

Create a custom generation pipeline plugin for AutoRAG-Research. Generation pipelines compose with a retrieval pipeline to implement RAG strategies — they retrieve relevant context and generate answers using an LLM.

## Architecture Overview

A generation plugin consists of two classes:

- **Config class** — extends `BaseGenerationPipelineConfig`, defines parameters and wires up the pipeline class
- **Pipeline class** — extends `BaseGenerationPipeline`, implements the retrieve-then-generate logic

The pipeline receives a retrieval pipeline instance via composition and interacts with the database via `GenerationPipelineService`. Results are stored as `ExecutorResult` rows.

### Class Hierarchy

```
BasePipelineConfig (config.py)
  └── BaseGenerationPipelineConfig (config.py)
        └── YourPipelineConfig

BasePipeline (pipelines/base.py)
  └── BaseGenerationPipeline (pipelines/generation/base.py)
        └── YourPipeline
```

### Required Methods

**Config class** (`BaseGenerationPipelineConfig` subclass):
- `get_pipeline_class()` — return the pipeline class type
- `get_pipeline_kwargs()` — return kwargs passed to the pipeline constructor

**Pipeline class** (`BaseGenerationPipeline` subclass):
- `_generate(query_id, top_k)` — async method that retrieves context and generates an answer
- `_get_pipeline_config()` — return a config dict for DB storage

### Key Attributes Available in Pipeline

- `self._llm` — LangChain `BaseLanguageModel` instance for text generation
- `self._retrieval_pipeline` — Composed retrieval pipeline instance
- `self._service` — `GenerationPipelineService` for DB operations

## Steps

### Step 1: Scaffold the plugin

```bash
autorag-research plugin create my_rag --type=generation
```

This creates a `my_rag_plugin/` directory with:
```
my_rag_plugin/
├── pyproject.toml
├── src/
│   └── my_rag_plugin/
│       ├── __init__.py
│       ├── pipeline.py
│       └── generation/
│           └── my_rag.yaml
└── tests/
    └── test_my_rag.py
```

### Step 2: Understand the generated skeleton

Open `src/my_rag_plugin/pipeline.py`. You'll see two classes:

**Config class:**
```python
@dataclass(kw_only=True)
class MyRagPipelineConfig(BaseGenerationPipelineConfig):
    # llm and retrieval_pipeline_name are inherited
    pipeline_type: PipelineType = field(default=PipelineType.GENERATION, init=False)

    def get_pipeline_class(self) -> type["MyRagPipeline"]:
        return MyRagPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        return {}
```

**Pipeline class:**
```python
class MyRagPipeline(BaseGenerationPipeline):
    def __init__(self, session_factory, name, llm, retrieval_pipeline, schema=None, **kwargs):
        super().__init__(session_factory, name, llm, retrieval_pipeline, schema)

    def _get_pipeline_config(self) -> dict[str, Any]:
        return {"type": "my_rag"}

    async def _generate(self, query_id, top_k) -> GenerationResult:
        raise NotImplementedError
```

### Step 3: Implement the generation logic

> **DO NOT add your own `asyncio.gather`, `asyncio.Semaphore`, or any concurrency control inside
> `_generate`.** The base pipeline's `run()` method already handles parallel execution of all
> queries via `run_with_concurrency_limit()` (semaphore + gather), controlled by the
> `max_concurrency` config parameter. Your `_generate` method is called once per query —
> just implement the retrieve-and-generate logic for that single query and return the result.

The `_generate` method is where your RAG strategy lives. Here's the typical pattern:

```python
from autorag_research.orm.service.generation_pipeline import GenerationResult


async def _generate(self, query_id: int | str, top_k: int) -> GenerationResult:
    # 1. Retrieve relevant chunks using the composed retrieval pipeline
    results = await self._retrieval_pipeline._retrieve_by_id(query_id, top_k)
    chunk_ids = [r["doc_id"] for r in results]
    chunk_contents = self._service.get_chunk_contents(chunk_ids)

    # 2. Get query text
    query_text = self._get_query_text(query_id)

    # 3. Build prompt
    context = "\n\n".join(chunk_contents)
    prompt = f"Context:\n{context}\n\nQuestion: {query_text}\n\nAnswer:"

    # 4. Generate answer (async LLM call)
    response = await self._llm.ainvoke(prompt)

    # 5. Return GenerationResult
    return GenerationResult(text=str(response.content))
```

`GenerationResult` is a dataclass with:
- `text: str` — the generated answer (required)
- `token_usage: dict | None` — optional token usage info
- `execution_time_ms: float | None` — optional execution time

**Adding custom parameters:**

```python
@dataclass(kw_only=True)
class MyRagPipelineConfig(BaseGenerationPipelineConfig):
    prompt_template: str = "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    pipeline_type: PipelineType = field(default=PipelineType.GENERATION, init=False)

    def get_pipeline_class(self) -> type["MyRagPipeline"]:
        return MyRagPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        return {"prompt_template": self.prompt_template}
```

Then accept in the constructor:

```python
class MyRagPipeline(BaseGenerationPipeline):
    def __init__(self, session_factory, name, llm, retrieval_pipeline,
                 schema=None, prompt_template="...", **kwargs):
        super().__init__(session_factory, name, llm, retrieval_pipeline, schema)
        self.prompt_template = prompt_template
```

### Step 4: Update the YAML config

Edit `src/my_rag_plugin/generation/my_rag.yaml`:

```yaml
_target_: my_rag_plugin.pipeline.MyRagPipelineConfig
description: "My custom RAG generation"
name: my_rag
retrieval_pipeline_name: bm25
llm: gpt-4o-mini
prompt_template: |
  Context:
  {context}

  Question:
  {query}

  Answer:
top_k: 10
batch_size: 128
max_concurrency: 8
max_retries: 3
retry_delay: 1.0
```

Important config fields inherited from `BaseGenerationPipelineConfig`:
- `llm` — LLM model string (auto-converted to a LangChain model instance)
- `retrieval_pipeline_name` — name of the retrieval pipeline to compose with

### Step 5: Write tests

Edit `tests/test_my_rag.py`:

```python
from unittest.mock import AsyncMock, MagicMock

from my_rag_plugin.pipeline import MyRagPipelineConfig, MyRagPipeline


def test_config():
    config = MyRagPipelineConfig(
        name="my_rag",
        llm=MagicMock(),
        retrieval_pipeline_name="bm25",
    )
    assert config.get_pipeline_class() is MyRagPipeline


def test_config_custom_params():
    config = MyRagPipelineConfig(
        name="my_rag",
        llm=MagicMock(),
        retrieval_pipeline_name="bm25",
        prompt_template="Custom: {context} Q: {query}",
    )
    kwargs = config.get_pipeline_kwargs()
    assert "prompt_template" in kwargs
```

For tests with LLM calls, use `langchain_core.language_models.FakeListLLM` to mock the LLM.

### Step 6: Install and register

```bash
cd my_rag_plugin
pip install -e .
# or: uv pip install -e .

cd ..
autorag-research plugin sync
```

### Step 7: Verify

Check that the config was copied:
```bash
ls configs/pipelines/generation/my_rag.yaml
```

Use it in your experiment config by referencing the pipeline name. The Executor will automatically inject the retrieval pipeline specified by `retrieval_pipeline_name`.

## Reference

### Key Files
- `autorag_research/config.py` — `BaseGenerationPipelineConfig`
- `autorag_research/pipelines/base.py` — `BasePipeline`
- `autorag_research/pipelines/generation/base.py` — `BaseGenerationPipeline`
- `autorag_research/orm/service/generation_pipeline.py` — `GenerationPipelineService`, `GenerationResult`
- `autorag_research/plugin_registry.py` — Entry point discovery

### Example Implementations
- `autorag_research/pipelines/generation/basic_rag.py` — Simple retrieve-then-generate
- `autorag_research/pipelines/generation/ircot.py` — Interleaving retrieval with chain-of-thought
- `autorag_research/pipelines/generation/et2rag.py` — Entity-aware RAG
- `autorag_research/pipelines/generation/main_rag.py` — Main RAG pipeline

### YAML Config Examples
- `configs/pipelines/generation/basic_rag.yaml`
- `configs/pipelines/generation/ircot.yaml`
- `configs/pipelines/generation/main_rag.yaml`
