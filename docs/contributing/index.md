# Contributing

Guidelines for contributing to AutoRAG-Research.

## Getting Started

1. Fork the repository
2. Set up [development environment](development-setup.md)
3. Create a feature branch
4. Make changes
5. Submit a pull request

## Code Style

- Python 3.10+ type hints
- Line length: 120 characters
- Ruff for linting and formatting
- ty for type checking

## Pull Request Guidelines

1. Write tests for new functionality
2. Update documentation if needed
3. Run `make check` before submitting
4. Keep PRs focused on single changes

## Areas for Contribution

| Area | Description |
|------|-------------|
| Pipelines | New retrieval/generation algorithms |
| Metrics | Additional evaluation metrics |
| Datasets | New dataset ingestors |
| Documentation | Improvements and examples |
| Bug fixes | Issue resolution |

## Related

- [Development Setup](development-setup.md) - Environment setup
- [Custom Pipeline Tutorial](../tutorial/custom-pipeline.md) - Adding pipelines
- [Custom Metric Tutorial](../tutorial/custom-metric.md) - Adding metrics
