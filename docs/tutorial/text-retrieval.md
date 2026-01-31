# Text Retrieval Benchmark

Run a text retrieval benchmark without generation (no LLM required).

## Download Dataset

```bash
autorag-research data restore beir scifact_openai-small
```

Downloads BEIR SciFact (300 queries, 5,183 documents).

## Create Experiment Config

```yaml
# configs/experiment.yaml
db_name: beir_scifact_test_openai_small

pipelines:
  retrieval:
    - bm25
  generation: []

metrics:
  retrieval:
    - recall
    - precision
    - ndcg
    - mrr
  generation: []
```

## Run

```bash
autorag-research run --config-name=experiment
```

## Expected Output

```
Pipeline: bm25
  Recall@10: 0.847
  Precision@10: 0.085
  NDCG@10: 0.712
  MRR@10: 0.634
```

## Recommended Datasets

| Dataset | Queries | Documents | Best For |
|---------|---------|-----------|----------|
| BEIR SciFact | 300 | 5,183 | Scientific claims |
| BEIR NFCorpus | 323 | 3,633 | Biomedical |
| MTEB | varies | varies | General text |

See [Text Datasets](../datasets/text/index.md) for all options.

## Recommended Metrics

| Metric | Measures |
|--------|----------|
| Recall@k | Coverage of ground truth |
| NDCG@k | Ranking quality |
| MRR | First relevant position |

See [Retrieval Metrics](../metrics/retrieval/index.md) for details.

## Next

- [Text RAG](text-rag.md) - Add generation
- [Custom Pipeline](custom-pipeline.md) - Implement your algorithm
