# MixedbreadAI

Reranking via Mixedbread AI's API.

## Overview

| Field | Value |
|-------|-------|
| Type | API |
| Provider | [Mixedbread AI](https://mixedbread.ai/) |
| Default Model | `mixedbread-ai/mxbai-rerank-large-v1` |
| Env Variable | `MIXEDBREAD_API_KEY` |

## Configuration

```yaml
_target_: autorag_research.rerankers.mixedbreadai.MixedbreadAIReranker
model_name: mixedbread-ai/mxbai-rerank-large-v1
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| model_name | str | `mixedbread-ai/mxbai-rerank-large-v1` | Model name |
| api_key | str | None | API key (or use env var) |
| batch_size | int | 64 | Batch size |

## Models

| Model | Description |
|-------|-------------|
| mxbai-rerank-large-v1 | Large, best quality |
| mxbai-rerank-base-v1 | Base, balanced |

## Usage

```python
from autorag_research.rerankers import MixedbreadAIReranker

reranker = MixedbreadAIReranker()
results = reranker.rerank("query", documents, top_k=5)
```
