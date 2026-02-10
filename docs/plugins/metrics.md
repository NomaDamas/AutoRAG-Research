# Metric Plugins

Metric plugins add custom evaluation metrics to AutoRAG-Research. There are two types:
**retrieval metrics** evaluate search quality, and **generation metrics** evaluate answer quality.

| Type | Entry Point Group | Base Config Class | Implementation |
|------|------------------|-------------------|----------------|
| Retrieval Metric | `autorag_research.metrics` | `BaseRetrievalMetricConfig` | Function-based |
| Generation Metric | `autorag_research.metrics` | `BaseGenerationMetricConfig` | Function-based |

Both types follow the same pattern: a standalone metric function paired with a dataclass config
that wraps it.

## Scaffold

Use the CLI to generate a starter plugin:

```bash
# Retrieval metric
autorag-research plugin create my_recall --type=metric_retrieval

# Generation metric
autorag-research plugin create my_bleu --type=metric_generation
```

This creates a project directory with the config class, metric function stub, YAML config,
`pyproject.toml`, and a basic test file.

## Retrieval Metric

A retrieval metric is a plain function that computes a score. The config class wraps it via
`get_metric_func()`.

```python
from collections.abc import Callable
from dataclasses import dataclass

from autorag_research.config import BaseRetrievalMetricConfig


def my_recall_metric(**kwargs) -> float:
    """Compute custom recall metric."""
    # Your metric logic here
    return score


@dataclass
class MyRecallMetricConfig(BaseRetrievalMetricConfig):
    """Configuration for custom recall metric."""

    def get_metric_func(self) -> Callable:
        return my_recall_metric
```

Key points:

- The metric is a standalone function, not a class method.
- The config class wraps the function and exposes it through `get_metric_func()`.
- `BaseRetrievalMetricConfig` automatically sets `metric_type = MetricType.RETRIEVAL`.
- `get_metric_name()` is inherited and returns the function name by default.

### Inherited Fields

`BaseMetricConfig` provides these fields to all metric configs:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `description` | `str` | `""` | Optional description |
| `metric_type` | `MetricType` | Auto-set | `RETRIEVAL` or `GENERATION` |

Override `get_metric_kwargs()` to pass extra arguments to the metric function at evaluation time.

## Generation Metric

Generation metrics follow the same pattern but extend `BaseGenerationMetricConfig`.

```python
from collections.abc import Callable
from dataclasses import dataclass

from autorag_research.config import BaseGenerationMetricConfig


def my_bleu_metric(**kwargs) -> float:
    """Compute custom BLEU metric."""
    # Your metric logic here
    return score


@dataclass
class MyBleuMetricConfig(BaseGenerationMetricConfig):
    """Configuration for custom BLEU metric."""

    def get_metric_func(self) -> Callable:
        return my_bleu_metric
```

The only difference is the base class: `BaseGenerationMetricConfig` sets
`metric_type = MetricType.GENERATION`.

## YAML Configuration

Each metric plugin ships a YAML config file in a subcategory directory.

Retrieval metric:

```yaml
# src/my_recall_plugin/retrieval/my_recall.yaml
_target_: my_recall_plugin.metric.MyRecallMetricConfig
description: "Custom recall metric"
```

Generation metric:

```yaml
# src/my_bleu_plugin/generation/my_bleu.yaml
_target_: my_bleu_plugin.metric.MyBleuMetricConfig
description: "Custom BLEU metric"
```

The `_target_` field must be the fully-qualified path to the config class.

## Entry Points

Register the plugin in `pyproject.toml` so AutoRAG-Research can discover it:

```toml
[project.entry-points."autorag_research.metrics"]
my_recall = "my_recall_plugin"
```

After editing `pyproject.toml`, reinstall the package (`pip install -e .`) and run
`autorag-research plugin sync` to copy configs into the project.

## Use in Experiment

Reference your metric by name in the experiment config:

```yaml
# configs/experiment.yaml
metrics:
  retrieval:
    - recall       # built-in
    - my_recall    # your plugin
  generation:
    - rouge        # built-in
    - my_bleu      # your plugin
```

The metric name matches the entry point key defined in `pyproject.toml`.

## Testing

Test that the config class instantiates correctly and returns a callable metric function:

```python
from my_recall_plugin.metric import MyRecallMetricConfig


def test_metric_config():
    config = MyRecallMetricConfig()
    func = config.get_metric_func()
    assert func is not None
    assert callable(func)
```

For integration tests that call real APIs or require data, use the `@pytest.mark.api` or
`@pytest.mark.data` markers.

## Next

- [Retrieval Pipeline](retrieval-pipeline.md) -- build a custom retrieval pipeline plugin
- [Generation Pipeline](generation-pipeline.md) -- build a custom generation pipeline plugin
- [Best Practices](best-practices.md) -- naming, security, and common pitfalls
