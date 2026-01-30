# ViDoRe

Visual Document Retrieval benchmark.

## Overview

| Field | Value |
|-------|-------|
| Modality | Multimodal (Images) |
| Generation GT | No |

## Description

ViDoRe (Visual Document Retrieval) evaluates retrieval systems on document images, including PDF pages, slides, and screenshots.

## Download

```bash
autorag-research data restore vidore <dataset_name>_<embedding_model>
```

## Ingest from Source

```bash
autorag-research ingest --name=vidore --extra dataset-name=<name> --embedding-model=colpali
```

## Best For

- Visual document understanding
- PDF page retrieval
- Multimodal embedding evaluation
