# Plugin CLI

Command group for plugin discovery and scaffolding.

## Overview

The `autorag-research plugin` command group manages plugin lifecycle:
creating new plugins from templates and syncing installed plugin configs
into your project.

## plugin create

Scaffolds a new plugin project with build config, skeleton code, YAML config, and tests.

### Synopsis

```bash
autorag-research plugin create NAME --type=TYPE
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `NAME` | Yes | Plugin name (lowercase letters, digits, underscores only. Must start with a letter.) |
| `--type`, `-t` | Yes | Plugin type: `retrieval`, `generation`, `metric_retrieval`, `metric_generation`, `ingestor` |

### Name Validation

Names must match `^[a-z][a-z0-9_]*$`.

| Valid | Invalid |
|-------|---------|
| `my_search` | `MySearch` |
| `es_retrieval` | `123plugin` |
| `custom_bm25` | `../evil` |

### Generated Structure

```
my_search_plugin/
├── pyproject.toml           # Build config with entry_points
├── src/
│   └── my_search_plugin/
│       ├── __init__.py
│       ├── pipeline.py      # Pipeline skeleton (or metric.py for metrics)
│       └── retrieval/       # Subcategory directory
│           └── my_search.yaml
└── tests/
    └── test_my_search.py
```

### Generated Files

**`pyproject.toml`** -- Hatchling build system, `autorag-research` dependency,
and entry_points registration under the appropriate group.

**`pipeline.py`** (or **`metric.py`**) -- Skeleton class inheriting the correct
base class with `NotImplementedError` stubs for required methods.

**YAML config** -- Hydra-style config with `_target_` pointing to the config
class.

**Test file** -- Basic config instantiation test.

### Examples

```bash
# Retrieval pipeline plugin
autorag-research plugin create my_search --type=retrieval

# Generation pipeline plugin
autorag-research plugin create my_rag --type=generation

# Retrieval metric plugin
autorag-research plugin create my_recall --type=metric_retrieval

# Generation metric plugin
autorag-research plugin create my_bleu --type=metric_generation

# Ingestor plugin
autorag-research plugin create my_dataset --type=ingestor
```

## plugin sync

Discovers installed plugins and copies their YAML configs into the local project.

### Synopsis

```bash
autorag-research plugin sync
```

### How It Works

1. Scans `autorag_research.pipelines` and `autorag_research.metrics` entry_points
2. Loads each plugin module and finds YAML config files
3. Copies YAMLs to `configs/pipelines/{subcategory}/` or `configs/metrics/{subcategory}/`

Existing files are never overwritten. Delete a file manually to re-sync it.

### Output

```
Copied 2 config(s):
  + pipelines/retrieval/es_search.yaml  (from plugin: elasticsearch)
  + metrics/retrieval/custom_recall.yaml  (from plugin: my_metrics)

Skipped 1 config(s) (already exist):
  = pipelines/retrieval/bm25_custom.yaml  (from plugin: custom_bm25)

Total: 2 copied, 1 skipped
```

## Ingestor Workflow

Ingestor plugins use decorator-based registration (`@register_ingestor`) instead of
Hydra YAML configs. No `plugin sync` step is needed.

```bash
# 1. Scaffold
autorag-research plugin create my_dataset --type=ingestor

# 2. Implement
cd my_dataset_plugin
# Edit src/my_dataset_plugin/ingestor.py

# 3. Test locally
pytest tests/

# 4. Install
pip install -e .
# or for uv users:
uv add --editable ./my_dataset_plugin

# 5. Run the ingestor
autorag-research ingest --name=my_dataset
```

## Pipeline / Metric Workflow

Full development lifecycle for a plugin:

```bash
# 1. Scaffold
autorag-research plugin create my_search --type=retrieval

# 2. Implement
cd my_search_plugin
# Edit src/my_search_plugin/pipeline.py

# 3. Test locally
pytest tests/

# 4. Install
pip install -e .
# or for uv users:
uv add --editable ./my_search_plugin

# 5. Sync configs
cd /path/to/your/project
autorag-research plugin sync

# 6. Use in experiment
# Edit configs/experiment.yaml to include my_search
autorag-research run --config-name=experiment
```
