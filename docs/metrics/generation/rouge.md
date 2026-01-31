# ROUGE

Recall-Oriented Understudy for Gisting Evaluation.

## Overview

| Field | Value |
|-------|-------|
| Type | Generation |
| Range | [0, 1] |
| Higher is better | Yes |

## Description

ROUGE measures n-gram recall - how many reference n-grams appear in the generated text. Originally designed for summarization evaluation.

## Configuration

```yaml
_target_: autorag_research.evaluation.metrics.generation.RougeConfig
rouge_type: rougeL
use_stemmer: true
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| rouge_type | str | `rougeL` | ROUGE variant |
| use_stemmer | bool | true | Apply stemming |

## ROUGE Variants

| Variant | Description |
|---------|-------------|
| rouge1 | Unigram overlap |
| rouge2 | Bigram overlap |
| rougeL | Longest common subsequence |
| rougeLSum | LCS over sentences |

## When to Use

Good for:

- Summarization tasks
- Content coverage evaluation
- Recall-focused assessment

Limitations:

- Doesn't capture semantic similarity
- Position-insensitive (except rougeL)
