# BRIGHT

Reasoning-intensive retrieval benchmark.

## Overview

| Field | Value |
|-------|-------|
| Modality | Text |
| Generation GT | No |
| HF Repository | `bright-dumps` |

## Description

BRIGHT (Benchmarking Retrieval for Generative and Reasoning-Intensive Transformations) focuses on queries that require reasoning beyond simple keyword matching.

## Download

```bash
autorag-research data restore bright <dataset_name>_<embedding_model>
```

## Ingest from Source

```bash
autorag-research ingest --name=bright --extra dataset-name=<name> --embedding-model=openai-small
```

## Best For

- Complex query understanding
- Reasoning-based retrieval
- Advanced semantic matching
