# SDS KoPub VDR

Korean public-document Visual Document Retrieval benchmark.

## Overview

| Field | Value |
|-------|-------|
| Modality | Multimodal (Images + OCR/Text) |
| Generation GT | No |
| HF Repository | `mteb/SDSKoPubVDRT2ITRetrieval` |
| Primary Key Type | `string` |
| License | CC BY-SA 4.0 |

## Description

SDS KoPub VDR is a Korean visual document retrieval benchmark built from real public/government documents. This ingestor uses the MTEB text-to-image retrieval version, which exposes a clean BEIR-style layout with `corpus`, `queries`, and `qrels` configs.

The dataset has three Hugging Face configs:

- `queries`: text retrieval queries
- `corpus`: page images plus extracted text
- `qrels`: query-to-page relevance judgments

## Ingest from Source

```bash
autorag-research ingest --name=sds_kopub_vdr --embedding-model=colpali
```

By default, qrels are mapped to image chunks. To evaluate text chunks or mixed image/text retrieval, set `qrels-mode`:

```bash
autorag-research ingest --name=sds_kopub_vdr --extra qrels-mode=mixed --embedding-model=colpali
```

## Best For

- Korean public-document visual retrieval
- Layout-, table-, chart-, and image-heavy document retrieval
- Comparing image-only, text-only, and mixed page retrieval
