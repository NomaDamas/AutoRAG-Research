# SemScore

Semantic similarity using embedding models.

## Overview

| Field | Value |
|-------|-------|
| Type | Generation |
| Range | [-1, 1] |
| Higher is better | Yes |

## Description

SemScore computes cosine similarity between sentence embeddings of generated and reference text. Uses configurable embedding models.

## Configuration

```yaml
_target_: autorag_research.evaluation.metrics.generation.SemScoreConfig
embedding_model: openai-small
truncate_length: 8192
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| embedding_model | str | required | Embedding model config name |
| truncate_length | int | 8192 | Max text length |

## When to Use

Good for:

- Semantic correctness evaluation
- When using same embeddings as retrieval
- Fast semantic comparison

Limitations:

- Depends on embedding model quality
- May miss fine-grained differences
