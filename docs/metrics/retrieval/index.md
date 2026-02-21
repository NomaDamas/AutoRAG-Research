# Retrieval Metrics

Metrics for evaluating document retrieval quality.

## Available Metrics

| Metric | Measures | When to Use |
|--------|----------|-------------|
| [Recall@k](recall.md) | Coverage | Ensure all relevant docs found |
| [Full Recall@k](full-recall.md) | Complete coverage | All evidence groups must be retrieved |
| [Precision@k](precision.md) | Relevance | Minimize irrelevant results |
| [F1@k](f1.md) | Balance | Trade-off recall and precision |
| [NDCG@k](ndcg.md) | Ranking | Order matters |
| [MRR](mrr.md) | First hit | Single answer tasks |
| [MAP](map.md) | Overall quality | Comprehensive evaluation |

## Common Parameters

All retrieval metrics use the top-k retrieved results compared against ground truth relevance judgments.

## Base Class

```python
from autorag_research.evaluation.metrics import BaseRetrievalMetricConfig
from dataclasses import dataclass


@dataclass
class MyMetricConfig(BaseRetrievalMetricConfig):
    def get_metric_func(self):
        return my_metric_function
```
