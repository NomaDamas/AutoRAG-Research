# CRAG

Comprehensive RAG Benchmark support for generation-oriented evaluation with provided web search results.

## Overview

| Field | Value |
|-------|-------|
| Modality | Text |
| Generation GT | Yes |
| Retrieval GT | No |
| Source | facebookresearch/CRAG Task 1/2 dev file |

## Description

CRAG (Comprehensive RAG Benchmark) pairs factual questions with gold answers and per-query web search results.
This ingestor currently supports the Task 1/2 development dataset and stores:

- queries as `Query.contents`
- `answer` + `alt_ans` as `generation_gt`
- each `search_results[*]` entry as a text chunk built from title, URL, snippet, last-modified time, and extracted HTML text

## Scope Notes

- `subset=dev` maps to CRAG `split=0`
- `subset=test` maps to CRAG `split=1`
- `subset=train` currently aliases the public dev split because the supported source file does not publish a separate train split
- ingesting both `subset=train` and `subset=dev` into the same database duplicates the same upstream examples under different IDs
- retrieval relevance labels are **not** created because CRAG search results are candidate context, not authoritative qrels
- `min_corpus_cnt` is ignored because CRAG examples already carry their own per-query search results

## Ingest from Source

```bash
autorag-research ingest --name=crag --embedding-model=openai-small
```

Optional split selection:

```bash
autorag-research ingest --name=crag --subset=dev --embedding-model=openai-small
```

## Best For

- generation-focused RAG evaluation
- answer-grounded benchmarking with realistic web context
- experiments that need per-query search results without assuming retrieval qrels
