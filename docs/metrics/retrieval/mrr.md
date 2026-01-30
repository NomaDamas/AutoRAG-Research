# MRR

Mean Reciprocal Rank - measures position of first relevant result.

## Overview

| Field | Value |
|-------|-------|
| Type | Retrieval |
| Range | [0, 1] |
| Higher is better | Yes |

## Formula

$$MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{rank_i}$$

Where $rank_i$ is the position of the first relevant document for query $i$.

## Interpretation

| Score | Meaning |
|-------|---------|
| 1.0 | First result always relevant |
| 0.5 | First relevant at position 2 on average |
| 0.33 | First relevant at position 3 on average |
| 0.0 | No relevant docs in results |

## When to Use

Use when users typically only care about the first relevant result:

- Q&A systems
- Fact lookup
- Single-answer queries

## Configuration

```yaml
_target_: autorag_research.evaluation.metrics.retrieval.MRRConfig
```
