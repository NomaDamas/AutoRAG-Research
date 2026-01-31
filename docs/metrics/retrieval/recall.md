# Recall

Measures how many relevant documents were retrieved.

## Overview

| Field | Value |
|-------|-------|
| Type | Retrieval |
| Range | [0, 1] |
| Higher is better | Yes |

## Formula

$$Recall@k = \frac{|Retrieved@k \cap Relevant|}{|Relevant|}$$

## Interpretation

| Score | Meaning |
|-------|---------|
| 1.0 | All relevant docs retrieved |
| 0.8 | 80% of relevant docs found |
| 0.5 | Half of relevant docs found |
| 0.0 | No relevant docs retrieved |

## When to Use

Use when finding all relevant documents is critical, such as:

- Legal document search
- Medical information retrieval
- Comprehensive research

## Configuration

```yaml
_target_: autorag_research.evaluation.metrics.retrieval.RecallConfig
```
