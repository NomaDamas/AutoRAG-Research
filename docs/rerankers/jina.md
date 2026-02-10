# Jina

Reranking via Jina AI's API.

## Overview

| Field | Value |
|-------|-------|
| Type | API |
| Provider | [Jina AI](https://jina.ai/) |
| Default Model | `jina-reranker-v2-base-multilingual` |
| Env Variable | `JINA_API_KEY` |

## Configuration

```yaml
_target_: autorag_research.rerankers.jina.JinaReranker
model_name: jina-reranker-v2-base-multilingual
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| model_name | str | `jina-reranker-v2-base-multilingual` | Jina model name |
| api_key | str | None | API key (or use env var) |
| batch_size | int | 64 | Batch size |

## Models

| Model | Description |
|-------|-------------|
| jina-reranker-v2-base-multilingual | Multilingual, balanced |
| jina-reranker-v1-base-en | English only |
| jina-reranker-v1-turbo-en | Fast, English |

## Usage

```python
from autorag_research.rerankers import JinaReranker

reranker = JinaReranker()
results = reranker.rerank("query", documents, top_k=5)
```
