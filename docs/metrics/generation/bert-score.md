# BERTScore

Semantic similarity using BERT embeddings.

## Overview

| Field | Value |
|-------|-------|
| Type | Generation |
| Range | [-1, 1] |
| Higher is better | Yes |

## Description

BERTScore computes token-level similarity using contextual embeddings from BERT. Captures semantic similarity beyond exact word matches.

## Configuration

```yaml
_target_: autorag_research.evaluation.metrics.generation.BertScoreConfig
lang: en
batch: 64
n_threads: 4
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| lang | str | `en` | Language code |
| batch | int | 64 | Batch size |
| n_threads | int | 4 | Number of threads |

## When to Use

Good for:

- Semantic similarity assessment
- Paraphrase detection
- Meaning preservation evaluation

Limitations:

- Computationally expensive
- Requires BERT model
- May not capture factual correctness
