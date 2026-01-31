# Custom Metric

Create your own evaluation metric.

## Retrieval Metric

```python
from autorag_research.evaluation.metrics import (
    BaseRetrievalMetricConfig,
    metric,
    MetricInput,
)
from dataclasses import dataclass


@metric(fields_to_check=["retrieval_gt", "retrieved_ids"])
def hit_rate(metric_input: MetricInput) -> float:
    """Returns 1 if any ground truth doc was retrieved, 0 otherwise."""
    gt_ids = set()
    for group in metric_input.retrieval_gt:
        gt_ids.update(group)

    retrieved = set(metric_input.retrieved_ids)
    return 1.0 if gt_ids & retrieved else 0.0


@dataclass
class HitRateConfig(BaseRetrievalMetricConfig):
    def get_metric_func(self):
        return hit_rate
```

## Generation Metric

```python
from autorag_research.evaluation.metrics import (
    BaseGenerationMetricConfig,
    metric_loop,
    MetricInput,
)
from dataclasses import dataclass


@metric_loop(fields_to_check=["generation_gt", "generated_texts"])
def exact_match(metric_inputs: list[MetricInput]) -> list[float]:
    """Returns 1 if generated text exactly matches any ground truth."""
    scores = []
    for inp in metric_inputs:
        generated = inp.generated_texts.strip().lower()
        matches = any(gt.strip().lower() == generated for gt in inp.generation_gt)
        scores.append(1.0 if matches else 0.0)
    return scores


@dataclass
class ExactMatchConfig(BaseGenerationMetricConfig):
    def get_metric_func(self):
        return exact_match
```

## Add Configuration

```yaml
# configs/metrics/retrieval/hit_rate.yaml
_target_: my_module.HitRateConfig
```

## Use in Experiment

```yaml
# configs/experiment.yaml
metrics:
  retrieval:
    - recall
    - hit_rate  # your metric
  generation:
    - rouge
    - exact_match  # your metric
```

## Next

- [Metrics](../metrics/index.md) - See existing implementations
- [Custom Pipeline](custom-pipeline.md) - Test algorithms
