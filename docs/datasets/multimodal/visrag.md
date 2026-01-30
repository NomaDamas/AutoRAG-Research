# VisRAG

Visual RAG benchmark.

## Overview

| Field | Value |
|-------|-------|
| Modality | Multimodal (Images) |
| Generation GT | No |
| HF Repository | `visrag-dumps` |

## Description

VisRAG provides datasets for evaluating visual RAG systems, focusing on document images that require visual understanding for accurate retrieval.

## Download

```bash
autorag-research data restore visrag <dataset_name>_<embedding_model>
```

## Ingest from Source

```bash
autorag-research ingest --name=visrag --extra dataset-name=<name> --embedding-model=colpali
```

## Best For

- Visual RAG evaluation
- Document image understanding
- Multimodal retrieval research
