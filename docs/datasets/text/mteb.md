# MTEB

Massive Text Embedding Benchmark retrieval tasks.

## Overview

| Field | Value |
|-------|-------|
| Modality | Text |
| Generation GT | No |
| HF Repository | `mteb-dumps` |
| Paper | [Muennighoff et al., 2023](https://arxiv.org/abs/2210.07316) |

## Description

MTEB provides a comprehensive evaluation of text embedding models across multiple tasks. The retrieval subset focuses on document retrieval performance.

## Download

```bash
autorag-research data restore mteb <dataset_name>_<embedding_model>
```

## Ingest from Source

```bash
autorag-research ingest --name=mteb --extra dataset-name=<name> --embedding-model=openai-small
```

## Best For

- Embedding model comparison
- General text retrieval
- Dense retrieval evaluation
