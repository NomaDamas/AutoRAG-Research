# autorag-research init

Download default configuration files.

## Synopsis

```bash
autorag-research init [options]
```

## Description

Downloads default configuration templates from GitHub to the local `configs/` directory. Creates the directory structure needed for experiments.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--config-path`, `-cp` | `./configs` | Target directory for configs |

## Examples

```bash
# Initialize with defaults
autorag-research init

# Initialize to custom location
autorag-research init --config-path=/path/to/configs
```

## Created Structure

```
configs/
├── experiment.yaml
├── pipelines/
│   ├── retrieval/
│   │   └── bm25.yaml
│   └── generation/
│       └── basic_rag.yaml
└── metrics/
    ├── retrieval/
    │   └── recall.yaml
    └── generation/
        └── rouge.yaml
```

## Related

- [run](run.md) - Run experiments using configs
- [First Steps Tutorial](../tutorial/first-steps.md) - Getting started
