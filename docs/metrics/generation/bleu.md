# BLEU

Bilingual Evaluation Understudy - measures n-gram precision.

## Overview

| Field | Value |
|-------|-------|
| Type | Generation |
| Range | [0, 1] |
| Higher is better | Yes |

## Description

BLEU measures how many n-grams in the generated text appear in the reference text. Originally designed for machine translation evaluation.

## Configuration

```yaml
_target_: autorag_research.evaluation.metrics.generation.BleuConfig
tokenize: default
smooth_method: exp
max_ngram_order: 4
effective_order: true
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| tokenize | str | `default` | Tokenization method |
| smooth_method | str | `exp` | Smoothing for zero counts |
| max_ngram_order | int | 4 | Maximum n-gram size |
| effective_order | bool | true | Use effective order |

## When to Use

Good for:

- Translation-style tasks
- Tasks requiring exact phrase matching
- Comparing against single reference

Limitations:

- Doesn't capture semantic similarity
- Penalizes paraphrases
- Requires exact n-gram matches
