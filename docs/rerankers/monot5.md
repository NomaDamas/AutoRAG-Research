# MonoT5

Sequence-to-sequence reranking via T5 models.

## Overview

| Field | Value |
|-------|-------|
| Type | Seq2Seq |
| Library | torch, transformers |
| Default Model | `castorini/monot5-3b-msmarco-10k` |
| Paper | [Nogueira et al., 2020](https://arxiv.org/abs/2003.06713) |

## Installation

```bash
pip install "autorag-research[gpu]"
# or
uv add "autorag-research[gpu]"
```

## Configuration

```yaml
_target_: autorag_research.rerankers.monot5.MonoT5Reranker
model_name: castorini/monot5-3b-msmarco-10k
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| model_name | str | `castorini/monot5-3b-msmarco-10k` | MonoT5 model name |
| max_length | int | 512 | Maximum input sequence length |
| device | str | None | Device (auto-detected) |
| batch_size | int | 64 | Batch size for multiple queries |

## How It Works

1. Format input as `"Query: {query} Document: {passage} Relevant:"`
2. Generate the first output token
3. Compute softmax over "true" and "false" token logits
4. Use probability of "true" as relevance score

## Models

| Model | Size | Description |
|-------|------|-------------|
| castorini/monot5-base-msmarco-10k | 220M | Fast |
| castorini/monot5-large-msmarco-10k | 770M | Balanced |
| castorini/monot5-3b-msmarco-10k | 3B | Best quality |

## Usage

```python
from autorag_research.rerankers import MonoT5Reranker

reranker = MonoT5Reranker(model_name="castorini/monot5-base-msmarco-10k")
results = reranker.rerank("What is RAG?", ["doc1", "doc2", "doc3"], top_k=2)
```
