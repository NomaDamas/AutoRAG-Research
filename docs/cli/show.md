# autorag-research show

List available resources.

## Synopsis

```bash
autorag-research show <resource>
```

## Resources

| Resource | Description |
|----------|-------------|
| `datasets` | Available datasets |
| `ingestors` | Dataset ingestors |
| `pipelines` | Configured pipelines |
| `metrics` | Available metrics |
| `databases` | PostgreSQL databases |

## Examples

```bash
# List ingestors
autorag-research show ingestors

# List databases
autorag-research show databases

# List metrics
autorag-research show metrics
```

## Output Examples

### show ingestors

```
Available Ingestors:
┌──────────────┬─────────────────────────────────────┐
│ Name         │ Description                         │
├──────────────┼─────────────────────────────────────┤
│ beir         │ BEIR benchmark datasets             │
│ mteb         │ MTEB retrieval tasks                │
│ ragbench     │ RAGBench benchmark                  │
│ vidorev3     │ ViDoRe V3 visual documents          │
└──────────────┴─────────────────────────────────────┘
```

### show databases

```
Available Databases:
┌────────────────────────────────────┐
│ Name                               │
├────────────────────────────────────┤
│ beir_scifact_test_openai_small     │
│ ragbench_covidqa_test_openai_small │
└────────────────────────────────────┘
```

## Related

- [ingest](ingest.md) - Ingest datasets
- [data restore](data.md) - Download pre-indexed datasets
