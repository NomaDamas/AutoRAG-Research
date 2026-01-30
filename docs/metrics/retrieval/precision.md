# Precision

Measures how many retrieved documents were relevant.

## Overview

| Field | Value |
|-------|-------|
| Type | Retrieval |
| Range | [0, 1] |
| Higher is better | Yes |

## Formula

$$Precision@k = \frac{|Retrieved@k \cap Relevant|}{k}$$

## Interpretation

| Score | Meaning |
|-------|---------|
| 1.0 | All retrieved docs relevant |
| 0.5 | Half of retrieved docs relevant |
| 0.1 | 1 in 10 retrieved docs relevant |
| 0.0 | No retrieved docs relevant |

## When to Use

Use when minimizing irrelevant results is important, such as:

- User-facing search interfaces
- Limited display space
- High-precision applications

## Configuration

```yaml
_target_: autorag_research.evaluation.metrics.retrieval.PrecisionConfig
```
