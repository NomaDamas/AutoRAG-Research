# RAGBench

RAG evaluation benchmark with generation ground truth.

## Overview

| Field | Value |
|-------|-------|
| Modality | Text |
| Generation GT | Yes |
| HF Repository | `ragbench-dumps` |

## Description

RAGBench provides datasets specifically designed for evaluating full RAG pipelines, including both retrieval and generation components. Unlike retrieval-only benchmarks, it includes expected answers for generation evaluation.

## Sub-datasets

| Name | Domain |
|------|--------|
| covidqa | COVID-19 Q&A |
| pubmedqa | Biomedical |
| techqa | Technical |

## Download

```bash
autorag-research data restore ragbench covidqa_openai-small
```

## Ingest from Source

```bash
autorag-research ingest --name=ragbench --extra config=covidqa --embedding-model=openai-small
```

## Best For

- Full RAG pipeline evaluation
- Generation quality assessment
- End-to-end benchmarking
