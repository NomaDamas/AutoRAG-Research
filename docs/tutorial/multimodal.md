# Multimodal Retrieval

Benchmark visual document retrieval (PDF pages, screenshots).

## Prerequisites

- GPU recommended for image embeddings
- Multimodal embedding model (ColPali)

## Download Dataset

```bash
autorag-research data restore vidorev3 arxivqa_colpali
```

ViDoRe v3 contains document images with queries.

## Key Differences from Text

| Aspect | Text | Multimodal |
|--------|------|------------|
| Documents | Plain text | Images (PDF pages) |
| Embeddings | Text models | Vision models (ColPali) |
| BM25 | Available | Not available |

## Create Experiment Config

```yaml
# configs/experiment.yaml
db_name: vidorev3_arxivqa_test_colpali

pipelines:
  retrieval:
    - colpali
  generation: []

metrics:
  retrieval:
    - recall
    - ndcg
  generation: []
```

## Run

```bash
autorag-research run --config-name=experiment
```

## Recommended Datasets

| Dataset | Description |
|---------|-------------|
| ViDoRe v3 | Document images, multiple domains |
| VisRAG | Visual RAG benchmark |

See [Multimodal Datasets](../datasets/multimodal/index.md) for all options.

## Next

- [Custom Dataset](custom-dataset.md) - Add your own images
- [Custom Pipeline](custom-pipeline.md) - New retrieval method
