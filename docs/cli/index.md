# CLI Reference

Command-line interface for AutoRAG-Research.

## Environment Variables

| Variable | Description |
|----------|-------------|
| POSTGRES_HOST | PostgreSQL host |
| POSTGRES_PORT | PostgreSQL port |
| POSTGRES_USER | PostgreSQL user |
| POSTGRES_PASSWORD | PostgreSQL password |

## Commands

::: mkdocs-typer2
 :module: autorag_research.cli.app
 :name: autorag-research

### Quick Cleanup Example

```bash
autorag-research drop database --db-name=beir_scifact_test_openai_small --yes
```
