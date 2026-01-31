# AutoRAG-Research

[![Release](https://img.shields.io/github/v/release/NomaDamas/AutoRAG-Research)](https://img.shields.io/github/v/release/NomaDamas/AutoRAG-Research)
[![Build status](https://img.shields.io/github/actions/workflow/status/NomaDamas/AutoRAG-Research/main.yml?branch=main)](https://github.com/NomaDamas/AutoRAG-Research/actions/workflows/main.yml?query=branch%3Amain)
[![Commit activity](https://img.shields.io/github/commit-activity/m/NomaDamas/AutoRAG-Research)](https://img.shields.io/github/commit-activity/m/NomaDamas/AutoRAG-Research)
[![License](https://img.shields.io/github/license/NomaDamas/AutoRAG-Research)](https://img.shields.io/github/license/NomaDamas/AutoRAG-Research)

Automate your RAG research with reproducible benchmarks.

## What is AutoRAG-Research?

A Python framework for:

- Running RAG benchmarks on standard datasets
- Evaluating retrieval and generation pipelines
- Comparing algorithms with reproducible metrics

## Quick Start

```bash
pip install autorag-research
docker-compose up -d
autorag-research data restore beir scifact_openai-small
autorag-research run --db-name=beir_scifact_test_openai_small
```

## Choose Your Path

| I want to... | Go to |
|--------------|-------|
| Run text retrieval benchmarks | [Text Retrieval Tutorial](tutorial/text-retrieval.md) |
| Run full RAG with generation | [Text RAG Tutorial](tutorial/text-rag.md) |
| Work with visual documents | [Multimodal Tutorial](tutorial/multimodal.md) |
| Use my own dataset | [Custom Dataset Tutorial](tutorial/custom-dataset.md) |
| Test my own pipeline | [Custom Pipeline Tutorial](tutorial/custom-pipeline.md) |
| Create my own metric | [Custom Metric Tutorial](tutorial/custom-metric.md) |

## Documentation

- [Learn](learn/index.md) - Core concepts and architecture
- [Tutorial](tutorial/index.md) - Step-by-step guides
- [Datasets](datasets/index.md) - Available benchmarks
- [Pipelines](pipelines/index.md) - Retrieval and generation algorithms
- [Metrics](metrics/index.md) - Evaluation measures
- [CLI Reference](cli/index.md) - Command-line usage
- [API Reference](reference/index.md) - Python API
