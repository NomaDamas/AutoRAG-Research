# Full Recall

Measures whether all relevant document groups were retrieved (all-or-nothing).

## Overview

| Field | Value |
|-------|-------|
| Type | Retrieval |
| Range | {0, 1} |
| Higher is better | Yes |

## Formula

$$FullRecall@k = \begin{cases} 1.0 & \text{if every GT group has at least one hit in Retrieved@k} \\ 0.0 & \text{otherwise} \end{cases}$$

The final score is the average of Full Recall across all queries.

## Interpretation

| Score | Meaning |
|-------|---------|
| 1.0 | All evidence groups retrieved |
| 0.0 | At least one evidence group missing |

Unlike standard [Recall](recall.md) which measures the *fraction* of ground truth groups found, Full Recall is a strict binary measure -- a query scores 1.0 only when **every** group is satisfied.

## When to Use

Use when complete evidence coverage is required, such as:

- Multi-hop question answering (all supporting facts must be retrieved)
- Fact verification requiring multiple evidence pieces
- Measuring the "success rate" of queries with full coverage

## Configuration

```yaml
_target_: autorag_research.evaluation.metrics.retrieval.FullRecallConfig
```
