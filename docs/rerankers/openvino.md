# OpenVINO

Intel hardware-optimized reranking via OpenVINO.

## Overview

| Field | Value |
|-------|-------|
| Type | HW-optimized |
| Library | [optimum-intel](https://huggingface.co/docs/optimum/intel/) |
| Default Model | `BAAI/bge-reranker-large` |

## Installation

```bash
pip install "autorag-research[openvino]"
# or
uv add "autorag-research[openvino]"
```

## Configuration

```yaml
_target_: autorag_research.rerankers.openvino.OpenVINOReranker
model_name: BAAI/bge-reranker-large
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| model_name | str | `BAAI/bge-reranker-large` | HuggingFace model name |
| max_length | int | 512 | Maximum input sequence length |
| batch_size | int | 64 | Batch size for multiple queries |

## How It Works

1. Auto-exports HuggingFace model to OpenVINO IR format
2. Runs inference using Intel OpenVINO runtime
3. Applies sigmoid activation to logits for scores

## Usage

```python
from autorag_research.rerankers import OpenVINOReranker

reranker = OpenVINOReranker()
results = reranker.rerank("What is RAG?", ["doc1", "doc2", "doc3"], top_k=2)
```

## When to Use

Good for:

- Intel CPU-based deployments
- Production environments without GPU
- Optimized inference on Intel hardware
