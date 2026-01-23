# CLAUDE.md

## Project Overview

AutoRAG-Research is a Python framework for automating RAG (Retrieval-Augmented Generation) research workflows.
It provides tools for data ingestion, pipeline execution, and evaluation metrics.
The user can download pre-ingested datasets and run pre-made RAG pipelines and evaluate it.
The pipelines can be customized. There are two types (retrieval/generation) pipelines.

## Common Commands

```bash
# Setup
make install              # Create venv, install deps, setup pre-commit hooks
uv sync --all-groups --all-extras  # Install all deps including optional (gpu, search)

# Code Quality
make check               # Run all checks (ruff, ty type checker, deptry)

# Testing (requires Docker)
make test                # Full test with Docker PostgreSQL lifecycle management
make test-only           # Run tests (assumes PostgreSQL container is running)
make test-data          # Run tests only marked as data
make test-full         # Run all tests including api/gpu/data marked tests

# Run single test
uv run pytest tests/path/to/test_file.py::test_function_name -v

# Docker
make docker-up           # Start PostgreSQL container
make docker-wait         # Wait for PostgreSQL readiness
make docker-down         # Stop container
make clean-docker        # Remove container and volumes

# Docs
make docs                # Build and serve docs locally
```

## Architecture

The codebase follows a layered architecture with **Generic Repository + Unit of Work + Service Layer** patterns:

```
Executor/Evaluator (config.py, executor.py, evaluator.py)
    ↓
Pipeline Layer (pipelines/)
    ↓
Service Layer (orm/service/) - Business logic
    ↓
Unit of Work (orm/uow/) - Transaction management
    ↓
Repository Layer (orm/repository/) - Data access (GenericRepository[T])
    ↓
ORM Models (orm/models/) - SQLAlchemy with pgvector
```

**Pipeline Types:**
- **Retrieval Pipelines** (`pipelines/retrieval/`) - Vector search, BM25, hybrid retrieval
  - Extend `BaseRetrievalPipeline`
  - Use `RetrievalPipelineService` + `RetrievalUnitOfWork`
  - Methods: `.retrieve(query, top_k)` for single-query, `.run()` for batch
- **Generation Pipelines** (`pipelines/generation/`) - LLM-based answer generation
  - Extend `BaseGenerationPipeline`
  - Use `GenerationPipelineService` + `GenerationUnitOfWork`
  - Compose with retrieval pipelines for flexible RAG strategies
  - Example: `NaiveRAGPipeline` (single retrieve + generate)

**Key Entry Points:**
- `executor.py` - Orchestrates pipeline execution and metric evaluation
- `config.py` - Configuration dataclasses (ExecutorConfig, BasePipelineConfig, BaseMetricConfig)
- `orm/schema_factory.py` - Dynamic schema creation with custom embedding dimensions via `create_schema(dim)`

## Database Patterns

- **Generic Repository:** Base CRUD in `orm/repository/base.py`, specialized repos extend it
- **Unit of Work:** Context managers in `orm/uow/` ensure transaction safety
- **Service Layer:** Business logic in `orm/service/`, uses UoW for data access
- **Vector Search:** Uses VectorChord with pgvector `Vector` type
- **Driver:** psycopg3 (imported as `psycopg`)
- **Async:** Use SQLAlchemy's greenlet-based bridging (`sqlalchemy.util.concurrency.greenlet_spawn`)

## Testing Guidelines

- Use `db_session` fixture from conftest.py (don't create new sessions)
- Test data is pre-seeded in `postgresql/db/init/002-seed.sql` - use it
- Test file structure mirrors package structure in `tests/`
- One test function = one function under test
- Clean up any data you add in tests

**Test Markers:**
```python
@pytest.mark.gpu       # Requires GPU
@pytest.mark.api       # Requires LLM/API calls (prefer mocks)
@pytest.mark.data      # Downloads external data
@pytest.mark.asyncio   # Async test
```

Prefer mocks over real API calls (use LlamaIndex MockLLM/MockEmbedding).

## Code Style

- Python 3.10+ type hints: use `list`, `dict`, `|` (not `typing.List`, `typing.Optional`)
- Line length: 120 characters
- Ruff for linting/formatting
- Type checking: ty

## Logging
- Alwasy use `logger = logging.getLogger("AutoRAG-Research")` to initialize logger
- Do not use print statements for logging. Or any other logging methods.

## Utility Functions (DRY!)

Before implementing common functionality, CHECK existing utils first:

- `autorag_research/util.py` - Core utilities (list conversion, async helpers, image processing)

DO NOT duplicate - reuse existing helpers!

## Key Database Tables

- **Document/Page/Chunk/ImageChunk** - Content storage with embeddings
- **Query** - Search queries with ground truth
- **RetrievalRelation** - Query-to-chunk relevance (composite key: query_id, group_index, group_order)
- **Pipeline/Metric** - Configuration storage
- **ChunkRetrievedResult** - Retrieval pipeline outputs (query → chunks with scores)
- **ExecutorResult** - Generation pipeline outputs (generation_result, token_usage, execution_time)
- **EvaluationResult** - Metric evaluation outputs

## AI Instructions

Detailed patterns and examples are in `/ai_instructions/`:
- `db_pattern.md` - Repository, UoW, Service patterns with code examples
- `db_schema.md` - Complete DBML schema
- `test_code_generation_instructions.md` - Testing conventions

## Strict Rules to follow

- **Code quality checks are automated via hooks** - `make check` runs automatically after file edits and on session end
- Generate the corresponding test file correctly when you add the code, and runs the test code before commit (`make test`)
- Do not use relative imports.
- Always use `uv sync --all-extras --all-groups` to install dependencies to avoid missing optional packages.
