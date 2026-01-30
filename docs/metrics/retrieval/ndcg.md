# NDCG

Normalized Discounted Cumulative Gain - measures ranking quality.

## Overview

| Field | Value |
|-------|-------|
| Type | Retrieval |
| Range | [0, 1] |
| Higher is better | Yes |

## Formula

$$DCG@k = \sum_{i=1}^{k} \frac{rel_i}{\log_2(i+1)}$$

$$NDCG@k = \frac{DCG@k}{IDCG@k}$$

Where IDCG is the ideal DCG (perfect ranking).

## Interpretation

| Score | Meaning |
|-------|---------|
| 1.0 | Perfect ranking |
| 0.8 | Good ranking |
| 0.5 | Moderate ranking |
| 0.0 | No relevant docs |

## When to Use

Use when the order of results matters, not just their presence:

- Search result ranking
- Recommendation systems
- Any task where position affects user experience

## Configuration

```yaml
_target_: autorag_research.evaluation.metrics.retrieval.NDCGConfig
```

## Reference

[Jarvelin & Kekalainen, 2002](https://dl.acm.org/doi/10.1145/582415.582418)
