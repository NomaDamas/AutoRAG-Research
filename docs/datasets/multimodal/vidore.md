# ViDoRe

Visual Document Retrieval benchmark (V1 QA datasets).

## Overview

| Field | Value |
|-------|-------|
| Modality | Multimodal (Images) |
| Generation GT | Yes (varies by dataset) |
| Structure | 1:1 query-to-image pairs |

## Description

ViDoRe (Visual Document Retrieval) V1 evaluates retrieval systems on document images, including PDF pages, slides, and screenshots. Each dataset contains query-image pairs where the task is to retrieve the relevant document image for a given query.

## Available Datasets

| Dataset | Rows | Answer Format | Language |
|---------|------|---------------|----------|
| `arxivqa_test_subsampled` | 500 | Multiple choice (A/B/C/D) | English |
| `docvqa_test_subsampled` | 500 | Held out (test set) | English |
| `infovqa_test_subsampled` | 500 | Held out (test set) | English |
| `tabfquad_test_subsampled` | 280 | None | French |
| `tatdqa_test` | 1663 | None | English |
| `shiftproject_test` | 1000 | List of strings | French |
| `syntheticDocQA_artificial_intelligence_test` | 1000 | List of strings | English |
| `syntheticDocQA_energy_test` | ~1000 | List of strings | English |
| `syntheticDocQA_government_reports_test` | ~1000 | List of strings | English |
| `syntheticDocQA_healthcare_industry_test` | ~1000 | List of strings | English |

## Download

```bash
autorag-research data restore vidore <dataset_name>_<embedding_model>
```

## Ingest from Source

```bash
# Ingest ArxivQA dataset with ColPali embeddings
autorag-research ingest -n vidore --extra dataset-name=arxivqa_test_subsampled --embedding-model=colpali

# Ingest DocVQA with custom query limit
autorag-research ingest -n vidore --extra dataset-name=docvqa_test_subsampled --query-limit=100 --embedding-model=colpali

# Skip embedding (ingest data only)
autorag-research ingest -n vidore --extra dataset-name=tatdqa_test --skip-embedding
```

## Best For

- Visual document understanding
- PDF page retrieval
- Multimodal embedding evaluation
- Document QA benchmarking

## Reference

- [ViDoRe Benchmark Collection](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d)
