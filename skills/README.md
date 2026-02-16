# AutoRAG-Research Plugin Development Skills

This directory contains agent skills that guide developers through creating AutoRAG-Research plugins. Each skill provides a step-by-step walkthrough for a specific plugin type.

## Available Skills

| Skill | Description | Use When |
|---|---|---|
| [create-retrieval-plugin](create-retrieval-plugin/SKILL.md) | Build a custom retrieval pipeline | You need a new search/retrieval strategy (e.g., Elasticsearch, ColBERT) |
| [create-generation-plugin](create-generation-plugin/SKILL.md) | Build a custom generation pipeline | You need a new RAG generation strategy (e.g., chain-of-thought RAG) |
| [create-metric-plugin](create-metric-plugin/SKILL.md) | Build a custom evaluation metric | You need a new retrieval or generation metric |
| [create-ingestor-plugin](create-ingestor-plugin/SKILL.md) | Build a custom data ingestor | You need to ingest a new dataset format |

## Plugin System Overview

AutoRAG-Research supports 4 plugin types distributed as separate Python packages:

- **Retrieval pipelines** — registered via `autorag_research.pipelines` entry point
- **Generation pipelines** — registered via `autorag_research.pipelines` entry point
- **Metrics** — registered via `autorag_research.metrics` entry point
- **Ingestors** — registered via `autorag_research.ingestors` entry point or `@register_ingestor` decorator

All pipeline/metric plugins follow the same lifecycle:

1. Scaffold with `autorag-research plugin create`
2. Implement the skeleton code
3. Write a YAML config
4. Write tests
5. Install with `pip install -e .`
6. Sync configs with `autorag-research plugin sync`

Ingestors use a decorator-based registration pattern instead.

## Quick Start

Pick the skill that matches your plugin type and follow its step-by-step guide.
