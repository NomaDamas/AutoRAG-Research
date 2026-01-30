# ViDoRe v2

Visual Document Retrieval benchmark version 2.

## Overview

| Field | Value |
|-------|-------|
| Modality | Multimodal (Images) |
| Generation GT | No |
| HF Repository | `vidorev2-dumps` |

## Description

ViDoRe v2 is an updated version of the Visual Document Retrieval benchmark with improved annotations and additional document types.

## Download

```bash
autorag-research data restore vidorev2 <dataset_name>_<embedding_model>
```

## Ingest from Source

```bash
autorag-research ingest --name=vidorev2 --extra dataset-name=<name> --embedding-model=colpali
```

## Best For

- Visual document understanding
- PDF page retrieval
- Multimodal embedding evaluation
