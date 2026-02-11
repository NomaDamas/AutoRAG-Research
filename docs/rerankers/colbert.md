# ColBERT

Token-level MaxSim reranking via ColBERT.

## Overview

| Field | Value |
|-------|-------|
| Type | Token-level |
| Library | torch, transformers |
| Default Model | `colbert-ir/colbertv2.0` |

## Installation

```bash
pip install "autorag-research[gpu]"
# or
uv add "autorag-research[gpu]"
```

## Configuration

```yaml
_target_: autorag_research.rerankers.colbert.ColBERTReranker
model_name: colbert-ir/colbertv2.0
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| model_name | str | `colbert-ir/colbertv2.0` | ColBERT model name |
| max_length | int | 512 | Maximum input sequence length |
| device | str | None | Device (auto-detected) |
| batch_size | int | 64 | Batch size for multiple queries |

## How It Works

1. Encode query and document into token embeddings
2. Compute cosine similarity matrix between all token pairs
3. For each query token, take the maximum similarity (MaxSim)
4. Average the MaxSim scores across query tokens

## Usage

```python
from autorag_research.rerankers import ColBERTReranker

reranker = ColBERTReranker()
results = reranker.rerank("What is RAG?", ["doc1", "doc2", "doc3"], top_k=2)
```
