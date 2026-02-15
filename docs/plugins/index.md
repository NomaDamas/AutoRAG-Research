# Plugins

Extend AutoRAG-Research with custom pipelines and metrics.

## Overview

AutoRAG-Research supports external plugins that add custom pipelines, metrics,
and data ingestors. Plugins use Python's `entry_points` mechanism for discovery.
Install a plugin package, run `plugin sync` to copy its configs (pipelines/metrics),
or use `autorag-research ingest` to run ingestor plugins directly.

## How It Works

1. Install a plugin package (`pip install autorag-research-elasticsearch`)
2. The package registers entry_points under `autorag_research.pipelines`, `autorag_research.metrics`, or `autorag_research.ingestors`
3. For pipelines/metrics: run `autorag-research plugin sync` to copy YAML configs into `configs/`
4. For ingestors: run `autorag-research ingest --name=<plugin_name>` to ingest data directly

## Plugin Types

| Type | Entry Point Group | Base Config Class | Base Pipeline/Metric Class |
|------|-------------------|-------------------|----------------------------|
| Retrieval Pipeline | `autorag_research.pipelines` | `BaseRetrievalPipelineConfig` | `BaseRetrievalPipeline` |
| Generation Pipeline | `autorag_research.pipelines` | `BaseGenerationPipelineConfig` | `BaseGenerationPipeline` |
| Retrieval Metric | `autorag_research.metrics` | `BaseRetrievalMetricConfig` | Function-based |
| Generation Metric | `autorag_research.metrics` | `BaseGenerationMetricConfig` | Function-based |
| Data Ingestor | `autorag_research.ingestors` | N/A (`@register_ingestor`) | `TextEmbeddingDataIngestor` |

## Quick Start

```bash
# Scaffold a new retrieval plugin
autorag-research plugin create my_search --type=retrieval

# Edit the generated code
cd my_search_plugin
# ... implement your logic in src/my_search_plugin/pipeline.py

# Install in development mode
pip install -e .

# Sync configs to your project
cd /path/to/your/project
autorag-research plugin sync
```

### Ingestor Plugin

```bash
# Scaffold a new ingestor plugin
autorag-research plugin create my_dataset --type=ingestor

# Edit the generated code
cd my_dataset_plugin
# ... implement your logic in src/my_dataset_plugin/ingestor.py

# Install in development mode
pip install -e .

# Run the ingestor
autorag-research ingest --name=my_dataset
```

## Next Steps

- [CLI Reference](cli.md)
- [Retrieval Pipeline](retrieval-pipeline.md)
- [Generation Pipeline](generation-pipeline.md)
- [Metrics](metrics.md)
- [Best Practices](best-practices.md)
