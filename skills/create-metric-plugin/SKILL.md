# Create Metric Plugin

## Description

Guide the developer through creating a custom evaluation metric plugin for AutoRAG-Research. Metrics evaluate the quality of retrieval or generation pipeline outputs. This skill covers both retrieval metrics and generation metrics.

## Architecture Overview

A metric plugin consists of:

- **Metric function** — a function that computes a score from a `MetricInput`
- **Config class** — extends `BaseRetrievalMetricConfig` or `BaseGenerationMetricConfig`, wires up the metric function

Metrics use the `@metric` or `@metric_loop` decorator from `autorag_research.evaluation.metrics.util` for input validation and batch processing.

### Class Hierarchy

```
BaseMetricConfig (config.py)
  ├── BaseRetrievalMetricConfig (config.py)
  │     └── YourRetrievalMetricConfig
  └── BaseGenerationMetricConfig (config.py)
        └── YourGenerationMetricConfig
```

### Required Methods

**Config class** (`BaseRetrievalMetricConfig` or `BaseGenerationMetricConfig` subclass):
- `get_metric_func()` — return the callable metric function
- `get_metric_kwargs()` — (optional) return additional kwargs for the metric function

### MetricInput Schema

All metric functions receive a `MetricInput` dataclass (`autorag_research/schema.py`):

```python
@dataclass
class MetricInput:
    query: str | None = None
    retrieval_gt: list[list[str]] | None = None          # Ground truth chunk IDs (AND/OR groups)
    retrieved_ids: list[str] | None = None                # Retrieved chunk IDs
    relevance_scores: dict[str, int] | None = None        # ID -> graded relevance
    retrieved_contents: list[str] | None = None            # Retrieved chunk texts
    retrieval_gt_contents: list[list[str]] | None = None   # Ground truth chunk texts
    generated_texts: str | None = None                     # Generated answer
    generation_gt: list[str] | None = None                 # Ground truth answers
    prompt: str | None = None                              # LLM prompt used
    generated_log_probs: list[float] | None = None         # Token log probabilities
```

### Metric Decorators

- **`@metric(fields_to_check=[...])`** — processes each `MetricInput` individually (one at a time)
- **`@metric_loop(fields_to_check=[...])`** — processes all `MetricInput`s at once (batch)

Both decorators validate that the specified fields are non-None before calling the function.

## Steps

### Step 1: Scaffold the plugin

For a retrieval metric:
```bash
autorag-research plugin create my_metric --type=metric_retrieval
```

For a generation metric:
```bash
autorag-research plugin create my_metric --type=metric_generation
```

This creates a `my_metric_plugin/` directory with:
```
my_metric_plugin/
├── pyproject.toml          # Entry point: autorag_research.metrics
├── src/
│   └── my_metric_plugin/
│       ├── __init__.py
│       ├── metric.py       # Metric function + config
│       └── retrieval/      # (or generation/)
│           └── my_metric.yaml
└── tests/
    └── test_my_metric.py
```

### Step 2: Understand the generated skeleton

Open `src/my_metric_plugin/metric.py`:

```python
from collections.abc import Callable
from dataclasses import dataclass

from autorag_research.config import BaseRetrievalMetricConfig  # or BaseGenerationMetricConfig


def my_metric_metric(**kwargs) -> float:
    raise NotImplementedError("Implement metric computation")


@dataclass
class MyMetricMetricConfig(BaseRetrievalMetricConfig):
    def get_metric_func(self) -> Callable:
        return my_metric_metric
```

### Step 3: Implement the metric function

#### Retrieval Metric Example

Use the `@metric` decorator for per-input processing:

```python
from autorag_research.evaluation.metrics.util import metric
from autorag_research.schema import MetricInput


@metric(fields_to_check=["retrieval_gt", "retrieved_ids"])
def my_retrieval_metric(metric_input: MetricInput) -> float:
    """Compute my custom retrieval metric.

    Args:
        metric_input: Contains retrieval_gt and retrieved_ids.

    Returns:
        Metric score as a float.
    """
    gt = metric_input.retrieval_gt   # list[list[str]] — AND/OR groups
    pred = metric_input.retrieved_ids  # list[str]

    # Your scoring logic here
    gt_flat = {item for group in gt for item in group}
    hits = sum(1 for p in pred if p in gt_flat)
    return hits / len(pred) if pred else 0.0
```

#### Generation Metric Example

```python
@metric(fields_to_check=["generation_gt", "generated_texts"])
def my_generation_metric(metric_input: MetricInput) -> float:
    gt = metric_input.generation_gt     # list[str] — ground truth answers
    pred = metric_input.generated_texts  # str — generated answer

    # Your scoring logic (e.g., exact match, overlap, etc.)
    return float(any(pred.strip().lower() == g.strip().lower() for g in gt))
```

#### Batch Metric Example

For metrics that need to process all inputs together (e.g., corpus-level scores), use `@metric_loop`:

```python
from autorag_research.evaluation.metrics.util import metric_loop


@metric_loop(fields_to_check=["generation_gt", "generated_texts"])
def my_batch_metric(metric_inputs: list[MetricInput], **kwargs) -> list[float]:
    """Process all inputs at once."""
    scores = []
    for mi in metric_inputs:
        # compute score for each input
        scores.append(0.0)
    return scores
```

### Step 4: Wire up the config class

```python
from collections.abc import Callable
from dataclasses import dataclass

from autorag_research.config import BaseRetrievalMetricConfig


@dataclass
class MyMetricConfig(BaseRetrievalMetricConfig):
    """Configuration for my custom metric."""

    def get_metric_func(self) -> Callable:
        return my_retrieval_metric
```

If your metric needs additional parameters:

```python
@dataclass
class MyMetricConfig(BaseRetrievalMetricConfig):
    threshold: float = 0.5

    def get_metric_func(self) -> Callable:
        return my_retrieval_metric

    def get_metric_kwargs(self) -> dict[str, Any]:
        return {"threshold": self.threshold}
```

### Step 5: Update the YAML config

Edit `src/my_metric_plugin/retrieval/my_metric.yaml`:

```yaml
_target_: my_metric_plugin.metric.MyMetricConfig
description: "My custom retrieval metric"
```

For a metric with extra parameters:

```yaml
_target_: my_metric_plugin.metric.MyMetricConfig
description: "My custom retrieval metric"
threshold: 0.5
```

### Step 6: Write tests

```python
from autorag_research.schema import MetricInput
from my_metric_plugin.metric import MyMetricConfig, my_retrieval_metric


def test_config():
    config = MyMetricConfig()
    assert config.get_metric_func() is not None


def test_metric_perfect_score():
    inputs = [MetricInput(
        retrieval_gt=[["doc1"], ["doc2"]],
        retrieved_ids=["doc1", "doc2"],
    )]
    scores = my_retrieval_metric(inputs)
    assert scores[0] == 1.0


def test_metric_zero_score():
    inputs = [MetricInput(
        retrieval_gt=[["doc1"]],
        retrieved_ids=["doc99"],
    )]
    scores = my_retrieval_metric(inputs)
    assert scores[0] == 0.0


def test_metric_none_fields():
    """Decorator returns None for inputs with missing required fields."""
    inputs = [MetricInput(retrieval_gt=None, retrieved_ids=None)]
    scores = my_retrieval_metric(inputs)
    assert scores[0] is None
```

### Step 7: Install and register

```bash
cd my_metric_plugin
pip install -e .

cd ..
autorag-research plugin sync
```

### Step 8: Verify

Check that the config was copied:
```bash
ls configs/metrics/retrieval/my_metric.yaml   # or metrics/generation/
```

Use it in your experiment config's `metrics` list.

## Reference

### Key Files
- `autorag_research/config.py` — `BaseMetricConfig`, `BaseRetrievalMetricConfig`, `BaseGenerationMetricConfig`
- `autorag_research/schema.py` — `MetricInput` dataclass
- `autorag_research/evaluation/metrics/util.py` — `@metric`, `@metric_loop` decorators
- `autorag_research/plugin_registry.py` — Entry point discovery

### Example Implementations
- `autorag_research/evaluation/metrics/retrieval.py` — Recall, Precision, F1, NDCG, MRR, MAP
- `autorag_research/evaluation/metrics/generation.py` — BLEU, ROUGE, BERTScore, SemScore

### YAML Config Examples
- `configs/metrics/retrieval/f1.yaml`
- `configs/metrics/retrieval/ndcg.yaml`
- `configs/metrics/generation/rouge.yaml`
