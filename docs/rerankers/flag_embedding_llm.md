# FlagEmbedding LLM

LLM-based reranking via BAAI FlagEmbedding.

## Overview

| Field | Value |
|-------|-------|
| Type | LLM-based |
| Library | [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) |
| Default Model | `BAAI/bge-reranker-v2-gemma` |

## Installation

```bash
pip install "autorag-research[gpu]"
# or
uv add "autorag-research[gpu]"
```

## Configuration

```yaml
_target_: autorag_research.rerankers.flag_embedding_llm.FlagEmbeddingLLMReranker
model_name: BAAI/bge-reranker-v2-gemma
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| model_name | str | `BAAI/bge-reranker-v2-gemma` | FlagEmbedding LLM model name |
| use_fp16 | bool | False | Use FP16 for inference |
| batch_size | int | 64 | Batch size for multiple queries |

## Usage

```python
from autorag_research.rerankers import FlagEmbeddingLLMReranker

reranker = FlagEmbeddingLLMReranker()
results = reranker.rerank("What is RAG?", ["doc1", "doc2", "doc3"], top_k=2)
```

## When to Use

Good for higher quality reranking when compute resources are available.
Uses LLM-scale models (e.g., Gemma) for more nuanced relevance scoring.
