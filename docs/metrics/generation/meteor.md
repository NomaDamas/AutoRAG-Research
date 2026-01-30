# METEOR

Metric for Evaluation of Translation with Explicit ORdering.

## Overview

| Field | Value |
|-------|-------|
| Type | Generation |
| Range | [0, 1] |
| Higher is better | Yes |

## Description

METEOR aligns generated text with reference using exact matches, stems, synonyms, and paraphrases. More flexible than BLEU for capturing semantic similarity.

## Configuration

```yaml
_target_: autorag_research.evaluation.metrics.generation.MeteorConfig
alpha: 0.9
beta: 3.0
gamma: 0.5
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| alpha | float | 0.9 | Recall weight |
| beta | float | 3.0 | Fragmentation penalty |
| gamma | float | 0.5 | Fragmentation weight |

## When to Use

Good for:

- Tasks allowing paraphrasing
- When synonyms should be rewarded
- More forgiving evaluation than BLEU

Limitations:

- Language-specific resources needed
- Slower than BLEU
