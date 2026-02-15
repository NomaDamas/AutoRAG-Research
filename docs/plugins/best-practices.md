# Best Practices

Guidelines, caveats, and common pitfalls for developing AutoRAG-Research plugins.

## Security

!!! warning "Security Note"
    Plugin discovery calls `ep.load()` which executes code from installed packages.
    Only install plugins from trusted sources. Review plugin code before installation.

- `plugin sync` loads plugin modules via `entry_points()` + `ep.load()` -- this runs arbitrary code from the installed package.
- Only install plugins from trusted, reviewed sources.
- Plugin names are validated against `^[a-z][a-z0-9_]*$` to prevent path traversal and injection.

## Plugin Naming

Plugin names must start with a lowercase letter and contain only lowercase letters, digits, and
underscores.

**Regex:** `^[a-z][a-z0-9_]*$`

| Name | Valid | Reason |
|------|-------|--------|
| `my_search` | Yes | |
| `es_retrieval` | Yes | |
| `custom_bm25` | Yes | |
| `MySearch` | No | Uppercase letters |
| `123plugin` | No | Starts with digit |
| `my-search` | No | Hyphens not allowed |
| `_private` | No | Starts with underscore |

## Package Layout

Use a nested layout with subcategory directories for YAML configs:

```
src/my_plugin/
├── __init__.py
├── pipeline.py          # or metric.py
└── retrieval/           # subcategory directory
    └── my_search.yaml
```

For ingestor plugins, no YAML config directory is needed:

```
src/my_dataset_plugin/
├── __init__.py
└── ingestor.py          # @register_ingestor decorated class
```

The subcategory directory determines where configs are synced (pipelines/metrics only):

- `retrieval/` syncs to `configs/pipelines/retrieval/` or `configs/metrics/retrieval/`
- `generation/` syncs to `configs/pipelines/generation/` or `configs/metrics/generation/`

Place the YAML file in the correct subcategory directory or it will not be discovered.

## Config Sync Behavior

- `plugin sync` **never overwrites** existing files. To re-sync a config, delete the existing file first.
- Configs are **copied**, not symlinked. Editing the local copy does not affect the plugin source.
- Install a plugin first, then run `plugin sync`. Order matters -- discovery requires the package to be installed.

## Testing Guidelines

- Use `MagicMock()` for LLM and `session_factory` in unit tests.
- Test config instantiation and abstract method implementations separately.
- Use pytest markers: `@pytest.mark.api` for tests needing real LLM calls.
- The scaffold includes a basic test file. Extend it with integration tests.

```python
from unittest.mock import MagicMock

import pytest


def test_pipeline_config():
    """Test config can be created and returns correct class."""
    config = MySearchPipelineConfig(name="test")
    assert config.get_pipeline_class() is MySearchPipeline
    assert "index_path" in config.get_pipeline_kwargs()


@pytest.mark.api
def test_pipeline_integration(db_session):
    """Integration test with real database (requires Docker)."""
    # Use db_session fixture from conftest.py
    pass
```

For ingestor plugins, use `FakeEmbeddings` from langchain_core:

```python
from langchain_core.embeddings import FakeEmbeddings


def test_ingestor_instantiation():
    """Test ingestor can be created with fake embeddings."""
    embeddings = FakeEmbeddings(size=128)
    ingestor = MyDatasetIngestor(
        embedding_model=embeddings,
        dataset_name="dataset_a",
    )
    assert ingestor.dataset_name == "dataset_a"
```

## Development Workflow

1. **Scaffold** -- `autorag-research plugin create NAME --type=TYPE`
2. **Implement** -- edit `pipeline.py` or `metric.py` with your logic
3. **Configure** -- edit the YAML config to set parameters
4. **Test** -- `pytest tests/` to run plugin tests
5. **Install** -- `pip install -e .` to install in dev mode
6. **Sync** -- `autorag-research plugin sync` to copy configs into the project
7. **Integrate** -- add the plugin name to your experiment config
8. **Run** -- `autorag-research run --config-name=experiment`

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Forgot `pip install -e .` | Plugin won't be discovered. Install before sync. |
| Config not appearing after sync | Check entry_points in `pyproject.toml`. Run `pip show your-plugin` to verify installation. |
| Existing config not updated | `plugin sync` never overwrites. Delete the old file and re-sync. |
| `_target_` path wrong | Must be fully-qualified: `package.module.ClassName` |
| LLM string not loading | Ensure the LLM provider package is installed (e.g., `langchain-openai`). |
| `get_pipeline_kwargs()` missing custom params | Only extra kwargs beyond `session_factory`, `name`, `schema` need to be returned. |

## See Also

- [Plugin Overview](index.md)
- [CLI Reference](cli.md)
- [Custom Pipeline Tutorial](../tutorial/custom-pipeline.md)
- [Custom Metric Tutorial](../tutorial/custom-metric.md)
