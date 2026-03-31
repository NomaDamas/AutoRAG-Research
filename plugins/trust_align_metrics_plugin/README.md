# trust-align-metrics-plugin

First-party AutoRAG-Research plugin that provides Trust-Align exact generation metrics.

## Metrics

- `trust_align_grounded_refusal_f1`
- `trust_align_answer_correctness_f1`

## Install (editable)

```bash
uv pip install -e plugins/trust_align_metrics_plugin
autorag-research plugin sync
```

The sync step copies plugin metric YAML files into `configs/metrics/generation/`.
