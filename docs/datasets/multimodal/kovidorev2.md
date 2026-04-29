# KoViDoRe v2

Korean Visual Document Retrieval benchmark version 2.

## Overview

| Field | Value |
|-------|-------|
| Modality | Multimodal (Images + Markdown text) |
| Generation GT | Yes |
| HF Repository | `whybe-choi/kovidore-v2-*-beir` |
| Primary Key Type | `bigint` |

## Description

KoViDoRe v2 is a Korean visual document retrieval benchmark from the `whybe-choi/kovidore-benchmark-beir-v2` collection with BEIR-style `corpus`, `queries`, and `qrels` subsets. Corpus rows contain page images plus markdown and layout metadata; query rows contain Korean questions and answer ground truth.

Supported domains and source datasets:

- `cybersecurity`: `whybe-choi/kovidore-v2-cybersecurity-beir`
- `economic`: `whybe-choi/kovidore-v2-economic-beir`
- `energy`: `whybe-choi/kovidore-v2-energy-beir`
- `hr`: `whybe-choi/kovidore-v2-hr-beir`

## Download

```bash
autorag-research data restore kovidorev2 <dataset_name>_<embedding_model>
```

## Ingest from Source

```bash
autorag-research ingest --name=kovidorev2 --extra dataset-name=hr --embedding-model=colpali
```

By default, qrels are mapped to image chunks. To evaluate text chunks or mixed image/text retrieval, set `qrels-mode`:

```bash
autorag-research ingest --name=kovidorev2 --extra dataset-name=hr qrels-mode=mixed --embedding-model=colpali
```

## Best For

- Korean visual document retrieval
- Multi-page visual reasoning
- Multimodal retrieval with generation answers
- Comparing image-only, text-only, and mixed page retrieval
