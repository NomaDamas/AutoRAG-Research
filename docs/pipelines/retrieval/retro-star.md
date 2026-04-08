# RETRO*

Wrap an existing retrieval pipeline, then rerank its candidates with rubric-based LLM scoring inspired by the RETRO* paper.

## Overview

| Field | Value |
|-------|-------|
| Type | Retrieval |
| Algorithm | Pointwise rubric-based reranking |
| Modality | Text |
| Paper | [Retro*: Optimizing LLMs for Reasoning-Intensive Document Retrieval](https://openreview.net/pdf?id=0WGl8PNMSA) |

## How It Works

1. Retrieve an initial candidate set from an existing retrieval pipeline such as BM25, vector search, or hybrid search.
2. Prompt an LLM with a task-specific relevance rubric for each query-document pair.
3. Parse the final `<score>` value from each reasoning trace.
4. Optionally sample multiple reasoning traces per pair and integrate their scores.
5. Return the highest-scoring documents as the wrapper pipeline output.

## Scope

This implementation covers the paper's practical inference-time reranking pattern only.

Out of scope for this MVP:

- the paper's SFT training stage
- the paper's RL optimization stage
- reproducing the published BRIGHT benchmark numbers end to end

## Configuration

```yaml
_target_: autorag_research.pipelines.retrieval.retro_star.RetroStarPipelineConfig
name: retro_star_bm25
llm: openai-gpt5-mini
retrieval_pipeline_name: bm25
candidate_top_k: 100
relevance_definition: >
  A document is relevant when it helps answer the query, including evidence
  that is indirect but still necessary for the required reasoning.
query_type: query
document_type: document
num_samples: 4
sample_weights: [0.1, 0.2, 0.3, 0.4]
top_k: 10
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| name | str | required | Unique pipeline instance name |
| llm | str | required | LLM config name (from `configs/llm/`) |
| retrieval_pipeline_name | str | required | Existing retrieval pipeline config to wrap |
| candidate_top_k | int | 100 | Number of wrapped candidates to rerank |
| relevance_definition | str | generic reasoning-aware definition | Rubric definition inserted into the prompt |
| query_type | str | `query` | Label used inside the prompt |
| document_type | str | `document` | Label used inside the prompt |
| num_samples | int | 1 | Number of reasoning traces to sample per candidate |
| sample_weights | list[float] \| null | null | Optional score-integration weights |
| max_document_tokens | int | 768 | Max candidate document tokens sent to the LLM |
| max_query_tokens | int | 256 | Max query tokens sent to the LLM |
| max_rerank_concurrency | int | 4 | Concurrent candidate-scoring calls per query |
| top_k | int | 10 | Final results per query |

## Default Prompt Behavior

The built-in prompt asks the model to:

1. analyze the query intent
2. analyze the candidate document
3. justify a 0-100 relevance score
4. end with the final score inside `<score>` tags

This mirrors the paper's rubric-based inference pattern while staying generic enough for arbitrary datasets.

## Usage

### Python API

```python
from langchain_openai import ChatOpenAI

from autorag_research.orm.connection import DBConnection
from autorag_research.pipelines.retrieval.bm25 import BM25RetrievalPipeline
from autorag_research.pipelines.retrieval.retro_star import RetroStarRetrievalPipeline

db = DBConnection.from_config()
session_factory = db.get_session_factory()

wrapped_retriever = BM25RetrievalPipeline(
    session_factory=session_factory,
    name="bm25",
    tokenizer="bert",
)

pipeline = RetroStarRetrievalPipeline(
    session_factory=session_factory,
    name="retro_star_bm25",
    llm=ChatOpenAI(model="gpt-5-mini"),
    retrieval_pipeline=wrapped_retriever,
    candidate_top_k=100,
    num_samples=4,
)
```

### YAML / Executor

```yaml
# configs/pipelines/retrieval/retro_star_bm25.yaml
_target_: autorag_research.pipelines.retrieval.retro_star.RetroStarPipelineConfig
name: retro_star_bm25
llm: openai-gpt5-mini
retrieval_pipeline_name: bm25
candidate_top_k: 100
num_samples: 4
```

The executor resolves `retrieval_pipeline_name`, instantiates the wrapped retriever, and injects it automatically.

## When to Use

Good for:

- reasoning-intensive benchmarks such as BRIGHT
- difficult queries where indirectly useful evidence matters
- comparing a stronger LLM-based reranking baseline against simpler retrievers

Consider other methods when:

- you need a lightweight retriever without LLM latency
- you want fully trained paper-faithful RETRO* checkpoints
- you only need sparse or dense retrieval without reranking
