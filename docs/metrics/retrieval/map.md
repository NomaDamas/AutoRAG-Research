# MAP

Mean Average Precision - comprehensive ranking metric.

## Overview

| Field | Value |
|-------|-------|
| Type | Retrieval |
| Range | [0, 1] |
| Higher is better | Yes |

## Formula

$$AP = \frac{1}{|Relevant|} \sum_{k=1}^{n} P(k) \cdot rel(k)$$

$$MAP = \frac{1}{|Q|} \sum_{q=1}^{|Q|} AP_q$$

Where $P(k)$ is precision at position $k$ and $rel(k)$ indicates if result $k$ is relevant.

## Interpretation

| Score | Meaning |
|-------|---------|
| 1.0 | All relevant docs at top positions |
| 0.8 | Good overall ranking |
| 0.5 | Moderate ranking quality |
| 0.0 | No relevant docs found |

## When to Use

Use for comprehensive evaluation of ranking quality across all queries:

- Benchmark comparisons
- System-level evaluation
- Research reporting

## Configuration

```yaml
_target_: autorag_research.evaluation.metrics.retrieval.MAPConfig
```
