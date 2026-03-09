# Trust-Align Metrics Plugin

This plugin provides Trust-Align exact generation metrics as a separate package:

- `trust_align_grounded_refusal_f1`
- `trust_align_answer_correctness_f1`

These metrics replace the former built-in `grounded_refusal_f1` and `answer_correctness_f1`.

## Install

```bash
uv pip install -e plugins/trust_align_metrics_plugin
```

If you use `pip`:

```bash
pip install -e plugins/trust_align_metrics_plugin
```

## Sync Plugin Configs

```bash
autorag-research plugin sync
```

After sync, metric YAML files are copied into `configs/metrics/generation/`.

## Use In Experiment Config

```yaml
metrics:
  generation:
    - trust_align_grounded_refusal_f1
    - trust_align_answer_correctness_f1
```

## Notes

- This plugin uses AutoAIS (`google/t5_xxl_true_nli_mixture`) for claim-evidence entailment.
- It requires additional dependencies (`torch`, `transformers`, `fuzzywuzzy`).
- Scores are returned on a percentage scale (0-100), following Trust-Align evaluator behavior.
