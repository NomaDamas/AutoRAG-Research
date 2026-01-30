# autorag-research run

Execute experiment pipelines with metric evaluation.

## Synopsis

```bash
autorag-research run [options]
```

## Options

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--config-name` | No | `experiment` | Config file name (without .yaml) |
| `--db-name` | No | from config | Database to use |
| `--verbose`, `-v` | No | - | Verbose output |

## Examples

```bash
# Run with default config
autorag-research run

# Run specific config
autorag-research run --config-name=my_experiment

# Override database
autorag-research run --db-name=beir_scifact_test_openai_small

# Verbose output
autorag-research run --config-name=experiment --verbose
```

## Config File Format

```yaml
# configs/experiment.yaml
db_name: beir_scifact_test_openai_small

pipelines:
  retrieval:
    - bm25
  generation:
    - basic_rag

metrics:
  retrieval:
    - recall
    - ndcg
    - mrr
  generation:
    - rouge
    - bleu
```

## Output

Results are stored in the database and printed to console:

```
Pipeline: bm25
  Recall@10: 0.847
  NDCG@10: 0.712
  MRR@10: 0.634

Pipeline: basic_rag
  ROUGE-L: 0.412
  BLEU: 0.287
```

## Related

- [init](init.md) - Download config templates
- [show databases](show.md) - List available databases
- [Text Retrieval Tutorial](../tutorial/text-retrieval.md) - Complete example
