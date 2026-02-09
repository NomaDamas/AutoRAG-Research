# FlashRank

Lightweight ONNX-based reranking.

## Overview

| Field | Value |
|-------|-------|
| Type | ONNX |
| Library | [flashrank](https://github.com/PrithivirajDamodaran/FlashRank) |
| Default Model | `ms-marco-MiniLM-L-12-v2` |

## Installation

```bash
pip install flashrank
```

## Configuration

```yaml
_target_: autorag_research.rerankers.flashrank.FlashRankReranker
model_name: ms-marco-MiniLM-L-12-v2
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| model_name | str | `ms-marco-MiniLM-L-12-v2` | FlashRank model name |
| max_length | int | 512 | Maximum input sequence length |
| batch_size | int | 64 | Batch size for multiple queries |

## Usage

```python
from autorag_research.rerankers import FlashRankReranker

reranker = FlashRankReranker()
results = reranker.rerank("What is RAG?", ["doc1", "doc2", "doc3"], top_k=2)
```

## When to Use

Good for:

- CPU-only environments
- Low-latency requirements
- Production deployments without GPU
