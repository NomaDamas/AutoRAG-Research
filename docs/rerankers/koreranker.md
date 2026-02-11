# KoReranker

Korean-specific document reranking.

## Overview

| Field | Value |
|-------|-------|
| Type | CrossEncoder (Korean) |
| Library | torch, transformers |
| Default Model | `Dongjin-kr/ko-reranker` |

## Installation

```bash
pip install "autorag-research[gpu]"
# or
uv add "autorag-research[gpu]"
```

## Configuration

```yaml
_target_: autorag_research.rerankers.koreranker.KoRerankerReranker
model_name: Dongjin-kr/ko-reranker
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| model_name | str | `Dongjin-kr/ko-reranker` | KoReranker model name |
| max_length | int | 512 | Maximum input sequence length |
| device | str | None | Device (auto-detected) |
| batch_size | int | 64 | Batch size for multiple queries |

## Usage

```python
from autorag_research.rerankers import KoRerankerReranker

reranker = KoRerankerReranker()
results = reranker.rerank("RAG란 무엇인가?", ["문서1", "문서2", "문서3"], top_k=2)
```

## When to Use

Specifically designed for Korean language documents. Use this when:

- Your corpus is primarily Korean text
- You need Korean-aware relevance scoring
