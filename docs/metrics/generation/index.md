# Generation Metrics

Metrics for evaluating text generation quality.

## Available Metrics

| Metric | Measures | When to Use |
|--------|----------|-------------|
| [BLEU](bleu.md) | N-gram overlap | Translation-style tasks |
| [METEOR](meteor.md) | Alignment | Better for paraphrases |
| [ROUGE](rouge.md) | N-gram recall | Summarization |
| [BERTScore](bert-score.md) | Semantic similarity | Meaning preservation |
| [BARTScore](bart-score.md) | Directional conditional likelihood | Faithfulness / precision / recall |
| [SemScore](sem-score.md) | Embedding similarity | Semantic correctness |
| [Response Relevancy](response-relevancy.md) | Question-answer alignment | RAGAS-style relevance checks |

Trust-Align exact refusal/correctness metrics are available as a plugin:
[Trust-Align Metrics Plugin](../../plugins/trust-align-metrics.md).

## Base Class

```python
from autorag_research.evaluation.metrics import BaseGenerationMetricConfig
from dataclasses import dataclass


@dataclass
class MyMetricConfig(BaseGenerationMetricConfig):
    def get_metric_func(self):
        return my_metric_function
```
