# VoyageAI

Reranking via Voyage AI's API.

## Overview

| Field | Value |
|-------|-------|
| Type | API |
| Provider | [Voyage AI](https://www.voyageai.com/) |
| Default Model | `rerank-2` |
| Env Variable | `VOYAGE_API_KEY` |

## Installation

```bash
pip install "autorag-research[reranker]"
# or
uv add "autorag-research[reranker]"
```

## Configuration

```yaml
_target_: autorag_research.rerankers.voyageai.VoyageAIReranker
model_name: rerank-2
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| model_name | str | `rerank-2` | Voyage model name |
| api_key | str | None | API key (or use env var) |
| batch_size | int | 64 | Batch size |

## Models

| Model | Description |
|-------|-------------|
| rerank-2 | Latest version |
| rerank-lite-1 | Lightweight, faster |

## Usage

```python
from autorag_research.rerankers import VoyageAIReranker

reranker = VoyageAIReranker()
results = reranker.rerank("query", documents, top_k=5)
```
