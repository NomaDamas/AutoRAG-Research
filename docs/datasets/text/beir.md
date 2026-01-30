# BEIR

Heterogeneous benchmark for information retrieval.

## Overview

| Field | Value |
|-------|-------|
| Modality | Text |
| Generation GT | No |
| HF Repository | `beir-dumps` |
| Paper | [Thakur et al., 2021](https://arxiv.org/abs/2104.08663) |

## Sub-datasets

| Name | Domain | Queries | Documents |
|------|--------|---------|-----------|
| scifact | Scientific | 300 | 5,183 |
| nfcorpus | Biomedical | 323 | 3,633 |
| fiqa | Financial | 648 | 57,638 |
| arguana | Argument | 1,406 | 8,674 |
| scidocs | Scientific | 1,000 | 25,657 |
| trec-covid | Biomedical | 50 | 171,332 |
| nq | Wikipedia | 3,452 | 2,681,468 |
| hotpotqa | Wikipedia | 7,405 | 5,233,329 |
| msmarco | Web | 6,980 | 8,841,823 |
| fever | Fact | 6,666 | 5,416,568 |
| climate-fever | Climate | 1,535 | 5,416,593 |
| dbpedia-entity | Wikipedia | 400 | 4,635,922 |
| quora | Questions | 10,000 | 522,931 |
| cqadupstack | StackExchange | 13,145 | 457,199 |

## Download

```bash
autorag-research data restore beir scifact_openai-small
```

## Ingest from Source

```bash
autorag-research ingest --name=beir --extra dataset-name=scifact --embedding-model=openai-small
```

## Best For

- Text retrieval benchmarking
- Zero-shot evaluation
- Sparse vs dense comparison
