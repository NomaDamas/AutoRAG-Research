# TART

Task-Aware Retrieval with Instructions.

## Overview

| Field | Value |
|-------|-------|
| Type | Instruction-T5 |
| Library | torch, transformers |
| Default Model | `facebook/tart-full-flan-t5-xl` |
| Paper | [Asai et al., 2022](https://arxiv.org/abs/2211.09260) |

## Installation

```bash
pip install "autorag-research[gpu]"
# or
uv add "autorag-research[gpu]"
```

## Configuration

```yaml
_target_: autorag_research.rerankers.tart.TARTReranker
model_name: facebook/tart-full-flan-t5-xl
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| model_name | str | `facebook/tart-full-flan-t5-xl` | TART model name |
| instruction | str | `Find passage to answer given question` | Task instruction |
| max_length | int | 512 | Maximum input sequence length |
| device | str | None | Device (auto-detected) |
| batch_size | int | 64 | Batch size for multiple queries |

## How It Works

1. Prepend task instruction to query: `"{instruction} [SEP] {query}"`
2. Encode instruction-query with document as input pair
3. Apply softmax to classification logits
4. Use positive class probability as relevance score

## Usage

```python
from autorag_research.rerankers import TARTReranker

reranker = TARTReranker(instruction="Find passage to answer given question")
results = reranker.rerank("What is RAG?", ["doc1", "doc2", "doc3"], top_k=2)
```
