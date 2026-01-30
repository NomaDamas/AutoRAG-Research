# MrTyDi

Multilingual retrieval benchmark.

## Overview

| Field | Value |
|-------|-------|
| Modality | Text |
| Generation GT | No |
| HF Repository | `mrtydi-dumps` |
| Paper | [Zhang et al., 2021](https://arxiv.org/abs/2108.08787) |

## Description

Mr. TyDi is a multilingual benchmark for retrieval, covering multiple languages with native speakers providing queries and relevance judgments.

## Languages

- Arabic
- Bengali
- English
- Finnish
- Indonesian
- Japanese
- Korean
- Russian
- Swahili
- Telugu
- Thai

## Download

```bash
autorag-research data restore mrtydi <language>_<embedding_model>
```

## Ingest from Source

```bash
autorag-research ingest --name=mrtydi --extra language=english --embedding-model=openai-small
```

## Best For

- Multilingual retrieval evaluation
- Cross-lingual transfer
- Non-English benchmarking
