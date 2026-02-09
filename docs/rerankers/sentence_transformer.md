# SentenceTransformer

Cross-encoder reranking via SentenceTransformers.

## Overview

| Field | Value |
|-------|-------|
| Type | CrossEncoder |
| Library | [sentence-transformers](https://sbert.net/) |
| Default Model | `cross-encoder/ms-marco-MiniLM-L-2-v2` |

## Installation

```bash
pip install sentence-transformers
```

## Configuration

```yaml
_target_: autorag_research.rerankers.sentence_transformer.SentenceTransformerReranker
model_name: cross-encoder/ms-marco-MiniLM-L-2-v2
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| model_name | str | `cross-encoder/ms-marco-MiniLM-L-2-v2` | HuggingFace model name |
| max_length | int | 512 | Maximum input sequence length |
| device | str | None | Device (auto-detected) |
| batch_size | int | 64 | Batch size for multiple queries |

## Models

| Model | Description |
|-------|-------------|
| cross-encoder/ms-marco-MiniLM-L-2-v2 | Fast, lightweight |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | Balanced |
| cross-encoder/ms-marco-MiniLM-L-12-v2 | Best quality |

## Usage

```python
from autorag_research.rerankers import SentenceTransformerReranker

reranker = SentenceTransformerReranker()
results = reranker.rerank("What is RAG?", ["doc1", "doc2", "doc3"], top_k=2)

for r in results:
    print(f"[{r.index}] {r.score:.3f}: {r.text[:50]}...")
```
