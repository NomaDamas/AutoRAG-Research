# FlagEmbedding

Cross-encoder reranking via BAAI FlagEmbedding.

## Overview

| Field | Value |
|-------|-------|
| Type | CrossEncoder |
| Library | [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) |
| Default Model | `BAAI/bge-reranker-large` |

## Installation

```bash
pip install "autorag-research[gpu]"
# or
uv add "autorag-research[gpu]"
```

## Configuration

```yaml
_target_: autorag_research.rerankers.flag_embedding.FlagEmbeddingReranker
model_name: BAAI/bge-reranker-large
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| model_name | str | `BAAI/bge-reranker-large` | FlagEmbedding model name |
| use_fp16 | bool | False | Use FP16 for inference |
| batch_size | int | 64 | Batch size for multiple queries |

## Models

| Model | Description |
|-------|-------------|
| BAAI/bge-reranker-large | Best quality |
| BAAI/bge-reranker-base | Balanced |
| BAAI/bge-reranker-v2-m3 | Multilingual |

## Usage

```python
from autorag_research.rerankers import FlagEmbeddingReranker

reranker = FlagEmbeddingReranker()
results = reranker.rerank("What is RAG?", ["doc1", "doc2", "doc3"], top_k=2)
```
