# Development Setup

Set up a development environment for contributing.

## Prerequisites

- Python 3.10+
- Docker
- Git
- uv (recommended) or pip

## Clone Repository

```bash
git clone https://github.com/NomaDamas/AutoRAG-Research.git
cd AutoRAG-Research
```

## Install Dependencies

```bash
# Using uv (recommended)
uv sync --all-extras --all-groups

# Using pip
pip install -e ".[dev,test]"
```

## Start PostgreSQL

```bash
docker-compose up -d
```

Wait for PostgreSQL to be ready:

```bash
make docker-wait
```

## Run Tests

```bash
# Full test suite
make test

# Quick tests (assumes PostgreSQL running)
make test-only

# Single test
uv run pytest tests/path/to/test_file.py::test_function_name -v
```

## Code Quality

```bash
# Run all checks
make check

# Individual tools
uv run ruff check .
uv run ruff format .
uv run ty
```

## Common Commands

| Command | Description |
|---------|-------------|
| `make install` | Create venv, install deps |
| `make check` | Run all code quality checks |
| `make test` | Full test with Docker lifecycle |
| `make test-only` | Run tests (PostgreSQL assumed running) |
| `make docs` | Build and serve docs locally |
| `make docker-up` | Start PostgreSQL container |
| `make docker-down` | Stop container |

## Test Markers

```python
@pytest.mark.gpu       # Requires GPU
@pytest.mark.api       # Requires LLM/API calls
@pytest.mark.data      # Downloads external data
@pytest.mark.asyncio   # Async test
```

## Directory Structure

```
autorag_research/
├── cli/                 # CLI commands
├── data/
│   └── ingestor/        # Dataset ingestors
├── evaluation/
│   └── metrics/         # Evaluation metrics
├── orm/                 # Database layer
│   ├── models/          # SQLAlchemy models
│   ├── repository/      # Data access
│   ├── service/         # Business logic
│   └── uow/             # Unit of Work
├── pipelines/
│   ├── retrieval/       # Retrieval pipelines
│   └── generation/      # Generation pipelines
├── config.py            # Configuration classes
├── executor.py          # Pipeline orchestration
└── util.py              # Utilities
```
