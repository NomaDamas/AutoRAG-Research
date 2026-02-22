---
name: create-metric-plugin
description: |
  Guide developers through creating a custom evaluation metric plugin for AutoRAG-Research.
  Covers both retrieval metrics (recall, precision, etc.) and generation metrics (BLEU, ROUGE, etc.).
  Walks through scaffolding, implementing metric functions with @metric decorators, writing configs,
  testing, and installing. Use when building a new evaluation metric.
allowed-tools:
  - Bash
  - Read
  - Write
  - Edit
---

# Create Metric Plugin

## Workflow

### 1. Scaffold

```bash
# For retrieval metric:
autorag-research plugin create my_metric --type=metric_retrieval

# For generation metric:
autorag-research plugin create my_metric --type=metric_generation
```

Read the generated `metric.py`, `pyproject.toml`, YAML config, and test file to understand the structure.

### 2. Implement the metric function

Use the `@metric` decorator (per-input) or `@metric_loop` decorator (batch) from `autorag_research.evaluation.metrics.util`. Both validate that required fields are non-None before calling.

- `@metric(fields_to_check=[...])` — function receives a single `MetricInput`, returns `float`
- `@metric_loop(fields_to_check=[...])` — function receives `list[MetricInput]`, returns `list[float]`

See `autorag_research/schema.py` for the full `MetricInput` dataclass definition.

### 3. Understanding `retrieval_gt` (AND/OR group structure)

For retrieval metrics, `metric_input.retrieval_gt` uses a **nested list structure** with AND/OR semantics:

```
retrieval_gt: list[list[str]]

Example: [["A", "B"], ["C"]]
  → Means: (A OR B) AND C
  → Each inner list is an OR group (any item satisfies the group)
  → Outer list is AND (ALL groups must be satisfied for complete retrieval)
```

This is critical for multi-hop queries where multiple evidence pieces are needed. Your metric must handle this structure correctly — don't just flatten it into a single set unless your metric semantics allow it.

**Examples:**
- `[["doc1"]]` — single required document
- `[["doc1", "doc2"], ["doc3"]]` — need (doc1 OR doc2) AND doc3
- `[["doc1"], ["doc2"], ["doc3"]]` — need doc1 AND doc2 AND doc3

See `retrieval_ndcg` in `autorag_research/evaluation/metrics/retrieval.py` for a real implementation that handles AND/OR groups with graded relevance.

### 4. Wire up config and install

The generated config class just needs `get_metric_func()` to return your metric function. If your metric takes extra kwargs, override `get_metric_kwargs()`.

```bash
cd my_metric_plugin
pip install -e .   # or: uv pip install -e .
cd .. && autorag-research plugin sync
```

Verify: `ls configs/metrics/retrieval/my_metric.yaml` (or `metrics/generation/`)

## Key Files

| Purpose | Path |
|---|---|
| Base config classes | `autorag_research/config.py` → `BaseRetrievalMetricConfig`, `BaseGenerationMetricConfig` |
| MetricInput schema | `autorag_research/schema.py` |
| Metric decorators | `autorag_research/evaluation/metrics/util.py` → `@metric`, `@metric_loop` |
| Plugin entry point discovery | `autorag_research/plugin_registry.py` |

## Examples

Study these existing implementations for patterns:

- `autorag_research/evaluation/metrics/retrieval.py` — Recall, Precision, F1, NDCG, MRR, MAP (all handle AND/OR groups)
- `autorag_research/evaluation/metrics/generation.py` — BLEU, ROUGE, BERTScore, SemScore
- YAML configs: `configs/metrics/retrieval/f1.yaml`, `configs/metrics/generation/rouge.yaml`
