# Generation Metrics

Metrics for evaluating text generation quality.

## Available Metrics

| Metric | Measures | When to Use |
|--------|----------|-------------|
| [BLEU](bleu.md) | N-gram precision | Translation-style tasks |
| [METEOR](meteor.md) | Alignment | Better for paraphrases |
| [ROUGE](rouge.md) | N-gram recall | Summarization |
| [BERTScore](bert-score.md) | Semantic similarity | Meaning preservation |
| [SemScore](sem-score.md) | Embedding similarity | Semantic correctness |
| [Response Relevancy](response-relevancy.md) | Question-answer alignment | RAGAS-style relevance checks |

## Base Class

```python
from autorag_research.evaluation.metrics import BaseGenerationMetricConfig
from dataclasses import dataclass


@dataclass
class MyMetricConfig(BaseGenerationMetricConfig):
    def get_metric_func(self):
        return my_metric_function
```
