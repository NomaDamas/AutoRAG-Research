# BARTScore

Directional semantic evaluation with a pretrained BART sequence-to-sequence model.

## Overview

| Field | Value |
|-------|-------|
| Type | Generation |
| Range | (-∞, 0] |
| Higher is better | Yes |

## Description

BARTScore treats evaluation as conditional text generation and scores the average
token log-probability of one text given another. This makes it useful for
direction-sensitive checks in RAG:

- `bart_score_faithfulness`: retrieved context → answer
- `bart_score_precision`: reference → answer
- `bart_score_recall`: answer → reference
- `bart_score_f1`: arithmetic mean of precision and recall

The implementation uses the paper's standard `facebook/bart-large-cnn`
checkpoint by default and keeps the metric deterministic.

## Configuration

```yaml
_target_: autorag_research.evaluation.metrics.generation.BartScoreFaithfulnessConfig
checkpoint: facebook/bart-large-cnn
batch_size: 4
max_length: 1024
device: auto
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| checkpoint | str | `facebook/bart-large-cnn` | Hugging Face BART checkpoint |
| batch_size | int | `4` | Pair scoring batch size |
| max_length | int | `1024` | Max tokenizer length for source and target |
| device | str | `auto` | `cuda`, `mps`, `cpu`, or automatic selection |

## Variant selection

| Variant | YAML | Required fields | Perspective |
|---------|------|-----------------|-------------|
| Faithfulness | `bart_score_faithfulness.yaml` | `retrieved_contents`, `generated_texts` | Does the answer follow the retrieved context? |
| Precision | `bart_score_precision.yaml` | `generation_gt`, `generated_texts` | How well is the answer supported by the reference? |
| Recall | `bart_score_recall.yaml` | `generation_gt`, `generated_texts` | How much reference content is covered by the answer? |
| F1 | `bart_score_f1.yaml` | `generation_gt`, `generated_texts` | Balanced semantic overlap |

When multiple references are available, AutoRAG keeps the best
per-example BARTScore for precision and recall before computing F1.

## When to Use

Good for:

- RAG faithfulness checks without an LLM judge
- Complementing BERTScore / AlignScore-style semantic metrics
- Separating support (`reference → answer`) from coverage (`answer → reference`)

Limitations:

- Slower than lexical metrics
- Requires a local BART checkpoint download on first use
- Scores are negative log-likelihoods, so raw values are less intuitive than bounded metrics
