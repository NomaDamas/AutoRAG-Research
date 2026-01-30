# Open-RAGBench

Open RAG Benchmark from arXiv PDFs by Vectara.

## Overview

| Field | Value |
|-------|-------|
| Modality | Text |
| Generation GT | Yes |
| HF Repository | `open-ragbench-dumps` |

## Description

Open-RAGBench is created from arXiv papers, providing a benchmark for RAG systems with both retrieval and generation ground truth.

## Download

```bash
autorag-research data restore open-ragbench <dataset_name>_<embedding_model>
```

## Ingest from Source

```bash
autorag-research ingest --name=open-ragbench --extra dataset-name=<name> --embedding-model=openai-small
```

## Best For

- Academic document retrieval
- Full RAG evaluation
- Scientific Q&A
