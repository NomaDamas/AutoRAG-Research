# Cohere

Reranking via Cohere's API.

## Overview

| Field | Value |
|-------|-------|
| Type | API |
| Provider | [Cohere](https://cohere.com/) |
| Default Model | `rerank-v3.5` |
| Env Variable | `COHERE_API_KEY` |

## Configuration

```yaml
_target_: autorag_research.rerankers.cohere.CohereReranker
model_name: rerank-v3.5
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| model_name | str | `rerank-v3.5` | Cohere model name |
| api_key | str | None | API key (or use env var) |
| batch_size | int | 64 | Batch size for multiple queries |

## Models

| Model | Description |
|-------|-------------|
| rerank-v3.5 | Latest, best quality |
| rerank-english-v3.0 | English optimized |
| rerank-multilingual-v3.0 | Multilingual |

## Usage

```python
from autorag_research.rerankers import CohereReranker

reranker = CohereReranker(model_name="rerank-v3.5")
results = reranker.rerank("What is RAG?", ["doc1", "doc2", "doc3"], top_k=2)

for r in results:
    print(f"[{r.index}] {r.score:.3f}: {r.text[:50]}...")
```
