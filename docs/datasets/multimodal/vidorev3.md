# ViDoRe v3

Visual Document Retrieval benchmark version 3.

## Overview

| Field | Value |
|-------|-------|
| Modality | Multimodal (Images) |
| Generation GT | No |
| HF Repository | `vidorev3-dumps` |

## Description

ViDoRe v3 is the latest version of the Visual Document Retrieval benchmark, featuring diverse document types across multiple domains.

## Sub-datasets

| Name | Domain |
|------|--------|
| arxivqa | Academic papers |
| docvqa | Document images |
| infovqa | Infographics |
| tabfquad | Tables |

## Download

```bash
autorag-research data restore vidorev3 arxivqa_colpali
```

## Ingest from Source

```bash
autorag-research ingest --name=vidorev3 --extra dataset-name=arxivqa --embedding-model=colpali
```

## Best For

- State-of-the-art visual retrieval evaluation
- Multi-domain document understanding
- ColPali model benchmarking
