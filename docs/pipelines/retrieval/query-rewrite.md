# Query Rewrite

Rewrite the query with an LLM, then retrieve with an existing retrieval pipeline.

## Overview

| Field | Value |
|-------|-------|
| Type | Retrieval |
| Algorithm | Rewrite-Retrieve |
| Modality | Text |
| Paper | [Query Rewriting in Retrieval-Augmented Large Language Models](https://aclanthology.org/2023.emnlp-main.322/) |

## How It Works

1. Receives a query
2. Uses an LLM to rewrite the query into a retrieval-optimized search query
3. Passes the rewritten query to a configured retrieval pipeline
4. Persists the wrapper pipeline's retrieval outputs as usual

This is different from HyDE:

- **Query Rewrite** changes the **query text** before retrieval.
- **HyDE** generates a **hypothetical answer passage**, embeds it, and searches with that embedding.

## Scope

This implementation covers the paper's practical inference-time rewrite-retrieve flow only.
The paper's trainable/RL rewriter is out of scope for this MVP.

## Configuration

```yaml
_target_: autorag_research.pipelines.retrieval.query_rewrite.QueryRewritePipelineConfig
name: query_rewrite_bm25
llm: openai-gpt5-mini
retrieval_pipeline_name: bm25
prompt_template: |
  Rewrite the following question into a concise search query.
  Keep the original intent, add missing retrieval hints only when helpful, and return only the rewritten query.

  Question: {query}
  Rewritten query:
top_k: 10
batch_size: 128
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| name | str | required | Unique pipeline instance name |
| llm | str | required | LLM config name (from `configs/llm/`) |
| retrieval_pipeline_name | str | required | Existing retrieval pipeline config to wrap |
| prompt_template | str | see below | Template with `{query}` placeholder |
| top_k | int | 10 | Results per query |
| batch_size | int | 128 | Queries per batch |

**Default prompt template:**

```text
Rewrite the following question into a concise search query.
Keep the original intent, add missing retrieval hints only when helpful, and return only the rewritten query.

Question: {query}
Rewritten query:
```

## Usage

### Python API

```python
from langchain_openai import ChatOpenAI

from autorag_research.orm.connection import DBConnection
from autorag_research.pipelines.retrieval.bm25 import BM25RetrievalPipeline
from autorag_research.pipelines.retrieval.query_rewrite import QueryRewriteRetrievalPipeline

db = DBConnection.from_config()
session_factory = db.get_session_factory()

wrapped_retriever = BM25RetrievalPipeline(
    session_factory=session_factory,
    name="bm25",
    tokenizer="bert",
)

pipeline = QueryRewriteRetrievalPipeline(
    session_factory=session_factory,
    name="query_rewrite_bm25",
    llm=ChatOpenAI(model="gpt-5-mini"),
    retrieval_pipeline=wrapped_retriever,
)

results = await pipeline.retrieve("Who wrote the original paper on transformers?", top_k=10)
```

### YAML / Executor

Create or reuse a wrapped retrieval config such as `configs/pipelines/retrieval/bm25.yaml`, then point the query rewrite config at it:

```yaml
# configs/pipelines/retrieval/query_rewrite_bm25.yaml
_target_: autorag_research.pipelines.retrieval.query_rewrite.QueryRewritePipelineConfig
name: query_rewrite_bm25
llm: openai-gpt5-mini
retrieval_pipeline_name: bm25
```

The executor will resolve `retrieval_pipeline_name`, instantiate the wrapped retriever, and inject it automatically.

## When to Use

Good for:

- conversational or underspecified queries
- preserving an existing retriever while improving query phrasing
- lightweight LLM-assisted retrieval baselines

Consider other methods when:

- you need the paper's trainable/RL rewriter
- you prefer dense retrieval over rewritten text search
- added LLM latency is unacceptable
