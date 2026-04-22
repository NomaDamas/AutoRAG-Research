# Metric Implementation Guide

This document describes the full checklist for adding a new evaluation metric to AutoRAG-Research.
Every step must be completed before a metric PR is considered merge-ready.

---

## 1. Metric Function

Write the metric function in the appropriate module:

- **Generation metrics** → `autorag_research/evaluation/metrics/generation.py`
- **Retrieval metrics** → `autorag_research/evaluation/metrics/retrieval.py`

### Signature & Decorator

Every metric function must follow this signature:

```python
def my_metric(metric_inputs: list[MetricInput], **kwargs) -> list[float]:
    ...
```

Use one of the two decorators from `autorag_research.evaluation.metrics.util`:

| Decorator | When to use |
|-----------|-------------|
| `@metric(fields_to_check=[...])` | Per-input metrics (function receives a single `MetricInput`) |
| `@metric_loop(fields_to_check=[...])` | Batch metrics (function receives the full `list[MetricInput]`) |

`fields_to_check` lists `MetricInput` field names that must be non-None for the metric to run
(e.g., `["generated_texts", "generation_gt"]`). Inputs that fail the check automatically get `None`.

### MetricInput fields reference

Defined in `autorag_research/schema.py`:

| Field | Type | Typical consumers |
|-------|------|-------------------|
| `query` | `str` | Response relevancy |
| `retrieval_gt_contents` | `list[list[str]]` | Faithfulness metrics |
| `retrieved_contents` | `list[str]` | Faithfulness metrics |
| `generated_texts` | `str` | All generation metrics |
| `generation_gt` | `list[str]` | Reference-based metrics (BLEU, ROUGE, BERTScore, etc.) |
| `retrieved_ids` / `retrieval_gt` | `list[str]` / `list[list[str]]` | Retrieval metrics |
| `relevance_scores` | `dict[str, int]` | Graded retrieval metrics (NDCG) |

---

## 2. Config Class

Every metric needs a `@dataclass` Config class in the same module as the function.

### Inheritance

| Metric type | Base class |
|-------------|------------|
| Generation | `BaseGenerationMetricConfig` (from `autorag_research.config`) |
| Retrieval | `BaseRetrievalMetricConfig` (from `autorag_research.config`) |

Both base classes inherit from `BaseMetricConfig` (ABC) and set `metric_type` automatically.

### Required methods

| Method | Purpose | When to override |
|--------|---------|-----------------|
| `get_metric_func()` | Return the callable metric function | **Always** (abstract) |
| `get_metric_name()` | Return metric name string for DB storage | Override if name ≠ function name |
| `get_metric_kwargs()` | Return `dict` of kwargs passed to the function | Override if metric has configurable params |
| `get_compute_granularity()` | Return `"query"` or `"dataset"` | Override only for dataset-level metrics |

### Dataclass fields = configurable parameters

Any parameter the user should be able to change via YAML becomes a dataclass field with a default value.
These fields are set by Hydra `instantiate()` when it reads the YAML `_target_` + fields.

### Example: simple metric (no kwargs)

```python
@dataclass
class ExactMatchConfig(BaseGenerationMetricConfig):
    def get_metric_func(self) -> Callable:
        return exact_match
```

### Example: metric with kwargs

```python
@dataclass
class RougeConfig(BaseGenerationMetricConfig):
    rouge_type: str = "rougeL"
    use_stemmer: bool = False
    split_summaries: bool = False

    def get_metric_name(self) -> str:
        return f"rouge_{self.rouge_type}"

    def get_metric_func(self) -> Callable:
        return rouge

    def get_metric_kwargs(self) -> dict[str, Any]:
        return {
            "rouge_type": self.rouge_type,
            "use_stemmer": self.use_stemmer,
            "split_summaries": self.split_summaries,
        }
```

### Example: shared base for a metric family

```python
@dataclass
class _BaseMyMetricConfig(BaseGenerationMetricConfig):
    checkpoint: str = "default-checkpoint"
    batch_size: int = 4

    def get_metric_kwargs(self) -> dict[str, Any]:
        return {"checkpoint": self.checkpoint, "batch_size": self.batch_size}

@dataclass
class MyMetricVariantAConfig(_BaseMyMetricConfig):
    def get_metric_func(self) -> Callable:
        return my_metric_variant_a
```

---

## 3. Module Exports

Add the metric function **and** Config class to both:

1. **`autorag_research/evaluation/metrics/{generation,retrieval}.py`** — defined here
2. **`autorag_research/evaluation/metrics/__init__.py`** — import and add to `__all__`

---

## 4. YAML Config File

Create a YAML file at the correct path:

- **Generation** → `configs/metrics/generation/<metric_name>.yaml`
- **Retrieval** → `configs/metrics/retrieval/<metric_name>.yaml`

### Required YAML structure

```yaml
_target_: autorag_research.evaluation.metrics.generation.MyMetricConfig
description: "Human-readable description of what this metric measures"
# All dataclass fields with their default (or custom) values:
param1: value1
param2: value2
```

- `_target_` must be the **fully-qualified class path** to the Config class.
- `description` is displayed by `discover_metrics()` in the CLI.
- List all configurable fields so users can see and override them.

### File naming convention

The YAML filename (without `.yaml`) becomes the metric's CLI-discoverable name.
Users reference it in `experiment.yaml`:

```yaml
metrics:
  generation: [rouge, my_metric]   # → configs/metrics/generation/my_metric.yaml
```

---

## 5. Tests

Tests go in `tests/autorag_research/evaluation/metrics/test_{generation,retrieval}.py`.

### Minimum required test coverage

| Test | What it verifies |
|------|-----------------|
| **Correctness test** | Metric function returns expected scores for known inputs |
| **Config wiring test** | `Config.get_metric_name()`, `get_metric_func()`, `get_metric_kwargs()` return correct values |
| **CLI discovery test** | `discover_metrics()` finds the new YAML (add assertion to existing `test_discover_metrics_finds_real_generation_configs`) |

### Optional but recommended

| Test | When |
|------|------|
| Edge case tests | Empty inputs, single reference vs. multi-reference, multilingual |
| Optional dependency guard test | If metric requires GPU/optional packages (`torch`, `transformers`) |
| Device resolution test | If metric has `device: auto` logic |

### Test data convention

- Use the shared test fixtures already defined at the top of the test module (`generation_gts`, `generations`, `similarity_generation_metric_inputs`, etc.).
- Use mocks/monkeypatch for expensive operations (model loading, GPU). Prefer `FakeListLLM` / `FakeEmbeddings` from `langchain_core`.
- Mark tests appropriately: `@pytest.mark.gpu`, `@pytest.mark.api`.

---

## 6. Optional Dependency Handling

If the metric depends on optional packages (e.g., `torch`, `transformers`):

1. **Do NOT import at module level.** Use lazy import inside the function or a dedicated `_import_*_runtime()` helper.
2. **Raise `ImportError` with actionable guidance** pointing to `uv sync --all-extras --all-groups` or the relevant extras group.
3. **Add a test** verifying the error message when the dependency is missing.

---

## 7. Pre-Merge Verification Checklist

Run these before opening a PR:

```bash
# Tests pass
uv run pytest tests/autorag_research/evaluation/metrics/test_generation.py -q

# Code quality
make check

# Metric is discoverable
uv run python -c "
from autorag_research.cli.utils import discover_metrics
from pathlib import Path
import autorag_research.cli as cli
cli.CONFIG_PATH = Path('configs').resolve()
print(discover_metrics('generation'))
"
```

---

## Quick Reference: File Touchpoints

| What | Where |
|------|-------|
| Metric function | `autorag_research/evaluation/metrics/{generation,retrieval}.py` |
| Config class | Same file as metric function |
| Module exports | `autorag_research/evaluation/metrics/__init__.py` |
| YAML config | `configs/metrics/{generation,retrieval}/<name>.yaml` |
| Tests | `tests/autorag_research/evaluation/metrics/test_{generation,retrieval}.py` |
| CLI discovery test | `tests/autorag_research/cli/test_utils.py` (add assertion) |
