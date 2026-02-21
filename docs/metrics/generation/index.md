# Generation Metrics

Metrics for evaluating text generation quality.

## Available Metrics

| Metric | Measures | When to Use |
|--------|----------|-------------|
| [BLEU](bleu.md) | N-gram overlap | Translation-style tasks |
| [METEOR](meteor.md) | Alignment | Better for paraphrases |
| [ROUGE](rouge.md) | N-gram recall | Summarization |
| [BERTScore](bert-score.md) | Semantic similarity | Meaning preservation |
| [SemScore](sem-score.md) | Embedding similarity | Semantic correctness |
| [Grounded Refusal F1](grounded-refusal-f1.md) | Refusal calibration on answerable/unanswerable split | Refusal behavior quality |
| [Answer Correctness F1](answer-correctness-f1.md) | Calibrated claim correctness over dataset | Paper-aligned answer correctness |

## Base Class

```python
from autorag_research.evaluation.metrics import BaseGenerationMetricConfig
from dataclasses import dataclass


@dataclass
class MyMetricConfig(BaseGenerationMetricConfig):
    def get_metric_func(self):
        return my_metric_function
```
