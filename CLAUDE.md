# CLAUDE.md - Instructions for Claude Code

This document provides comprehensive context about the AutoRAG-Research project to help Claude Code assist with development.

## Project Overview

**AutoRAG-Research** is a Python framework for automating RAG (Retrieval-Augmented Generation) research experiments.
Many RAG methods (including Multi-Modal RAG) and pre-built datasets are supported off-the-shelf.

It provides:

- A structured database schema for storing documents, chunks, queries, and evaluation results
- Pipeline abstractions for retrieval and generation experiments
- Evaluation metrics for retrieval (Recall, Precision, F1, NDCG, MRR, MAP) and generation (BLEU, ROUGE, BERTScore)
- Multi-vector embedding support for late interaction models (ColPali, ColBERT)
- An Executor that orchestrates pipeline execution with retry logic and metric evaluation

**Tech Stack:**
- Python 3.10+
- PostgreSQL 18 with VectorChord extension (for vector search)
- SQLAlchemy 2.0+ ORM
- PostgreSQL + VectorChord for vector storage
- psycopg3 (NOT psycopg2) as PostgreSQL driver
- LlamaIndex for LLM/embedding integrations
- uv for package management
- pytest for testing

---

## Project Structure

```
AutoRAG-Research/
├── autorag_research/          # Main package
│   ├── __init__.py
│   ├── config.py              # Pipeline/Metric configuration base classes
│   ├── evaluator.py           # Deprecated evaluator (use executor instead)
│   ├── executor.py            # Main Executor class for running experiments
│   ├── exceptions.py          # Custom exception classes
│   ├── schema.py              # Pydantic MetricInput dataclass
│   ├── util.py                # Utility functions
│   │
│   ├── data/                  # Dataset loaders (BEIR, ViDoRe)
│   │   ├── base.py            # Base dataset class
│   │   ├── beir.py            # BEIR dataset loader
│   │   ├── vidore.py          # ViDoRe dataset loader
│   │   ├── restore.py         # Database restore utilities
│   │   └── util.py            # Data utilities
│   │
│   ├── embeddings/            # Multi-vector embedding models
│   │   ├── base.py            # Base embedding classes
│   │   ├── colpali.py         # ColPali implementation
│   │   └── bipali.py          # BiPali implementation
│   │
│   ├── evaluation/            # Evaluation metrics
│   │   └── metrics/
│   │       ├── retrieval.py   # Retrieval metrics (recall, precision, ndcg, etc.)
│   │       ├── generation.py  # Generation metrics (bleu, rouge, bertscore)
│   │       └── util.py        # Metric utilities & decorators
│   │
│   ├── nodes/                 # Pipeline nodes (retrieval, reranking, query expansion)
│   │   ├── retrieval/
│   │   │   └── bm25.py        # BM25 retrieval node
│   │   ├── reranker/
│   │   └── query_expansion/
│   │
│   ├── orm/                   # Database layer (Repository + UoW pattern)
│   │   ├── schema.py          # ORM schema with default 768-dim embeddings
│   │   ├── schema_factory.py  # Factory for custom embedding dimensions
│   │   ├── types.py           # Custom SQLAlchemy types (VectorArray)
│   │   ├── util.py            # ORM utilities
│   │   │
│   │   ├── models/            # Additional ORM models
│   │   │   └── retrieval_gt.py
│   │   │
│   │   ├── repository/        # Repository layer (data access)
│   │   │   ├── base.py        # GenericRepository, BaseVectorRepository
│   │   │   ├── chunk.py       # ChunkRepository
│   │   │   ├── query.py       # QueryRepository
│   │   │   ├── document.py    # DocumentRepository
│   │   │   └── ...            # Other repositories
│   │   │
│   │   ├── uow/               # Unit of Work pattern
│   │   │   ├── base.py        # BaseUnitOfWork
│   │   │   ├── text_uow.py    # TextOnlyUnitOfWork
│   │   │   ├── multi_modal_uow.py  # MultiModalUnitOfWork
│   │   │   ├── retrieval_uow.py    # RetrievalUnitOfWork
│   │   │   └── evaluation_uow.py   # EvaluationUnitOfWork
│   │   │
│   │   └── service/           # Service layer (business logic)
│   │       ├── base.py        # BaseService
│   │       ├── text_ingestion.py
│   │       ├── multi_modal_ingestion.py
│   │       ├── retrieval_pipeline.py
│   │       ├── retrieval_evaluation.py
│   │       └── generation_evaluation.py
│   │
│   └── pipelines/             # Pipeline implementations
│       ├── base.py            # BasePipeline
│       ├── retrieval/
│       │   ├── base.py        # BaseRetrievalPipeline
│       │   └── bm25.py        # BM25RetrievalPipeline
│       └── generation/
│
├── tests/                     # Test files (mirrors package structure)
│   ├── conftest.py            # pytest fixtures (db_session, session_factory)
│   ├── mock.py                # Mock utilities
│   └── autorag_research/      # Tests mirror package structure
│
├── postgresql/                # PostgreSQL configuration
│   ├── docker-compose.yml     # Docker setup for VectorChord PostgreSQL
│   ├── .env                   # PostgreSQL environment variables
│   └── db/init/
│       ├── 001-schema.sql     # Database schema DDL
│       └── 002-seed.sql       # Test seed data
│
├── ai_instructions/           # AI/Claude instructions
│   ├── db_schema.md           # Database schema diagram (DBML format)
│   ├── db_pattern.md          # Repository + UoW pattern documentation
│   └── test_code_generation_instructions.md  # Test writing guidelines
│
├── docs/                      # MkDocs documentation
├── scripts/                   # Utility scripts
├── Makefile                   # Build/test commands
├── pyproject.toml             # Project configuration
└── uv.lock                    # Dependency lock file
```

---

## Database Schema

The database schema is defined in `postgresql/db/init/001-schema.sql` and the ORM models are in `autorag_research/orm/schema_factory.py`.

### Core Tables

The DB schema is crucial because it represents the document structure in RAG context.
Here are the descriptions about each table.
If you are going to ingest the new dataset, be aware of these tables' descriptions.

| Table | Description                                                                                                                               |
|-------|-------------------------------------------------------------------------------------------------------------------------------------------|
| `file` | File level (PDF, images). If dataset contains filename, use this field to store it. It is only for connecting real file.                  |
| `document` | Document is the group of the page. One file often treated as one document. This is the largest content size of the schema.                |
| `page` | Document pages with optional image content. Some document do not contain the page, like markdown document. Then, you can use single page. |
| `caption` | Text captions extracted from pages. Usually used OCR or etc.                                                                              |
| `chunk` | Text chunks with single/multi-vector embeddings. This is the atomic unit for vector/keyword search with user query.                       |
| `image_chunk` | Image chunks with embeddings for multi-modal retrieval. This is the atomic unit for search by image retrieval.                            |
| `query` | Evaluation queries with ground truth                                                                                                      |
| `retrieval_relation` | Ground truth: query -> chunk/image_chunk relations. a.k.a Retrieval Ground Truth                                                          |
| `pipeline` | Pipeline configurations (name, config JSON)                                                                                               |
| `metric` | Metric definitions (name, type)                                                                                                           |
| `executor_result` | Pipeline execution results per query including generation results from the generation pipeline.                                           |
| `evaluation_result` | Metric evaluation results                                                                                                                 |
| `chunk_retrieved_result` | Retrieved chunks per query/pipeline. Results from the retrieval pipeline.                                                                 |
| `image_chunk_retrieved_result` | Retrieved image chunks per query/pipeline. Results from the retrieval pipeline.                                                           |
| `summary` | Aggregated metrics per pipeline. For getting statistics from the result.                                                                  |

Most of the datasets only have `chunk` or `image_chunk` as the atomic unit.
You have to make sure that 'searchable unit' in the dataset is only `chunk` or `image_chunk`.
`Document`, `Page`, and `Caption` can be used for the parent-child document retrieval, or etc.
It is available if the dataset supported it.

### Schema Reference

The schema uses DBML format and is documented in `ai_instructions/db_schema.md`.

### Vector Columns

- `embedding`: Single vector (e.g., `VECTOR(768)`) for standard retrieval
- `embeddings`: Multi-vector array (e.g., `VECTOR(768)[]`) for late interaction models like ColPali/ColBERT

### Creating Custom Schemas

```python
# Default 768-dimensional embeddings
from autorag_research.orm.schema import Base, Chunk, Query

# Custom dimension
from autorag_research.orm.schema_factory import create_schema
schema = create_schema(1024)  # 1024-dimensional embeddings
# You need to detect embedding dimension dynamically when ingesting.
# Use: schema.Base, schema.Chunk, schema.Query, etc.

# String primary keys instead of bigint
# The string primary key can be used if the dataset uses string IDs.
schema = create_schema(768, primary_key_type="string")
```

---

## Architecture Patterns

### Generic Repository + Unit of Work + Service Layer

The codebase follows these design patterns for database operations:

#### Repository Pattern (`autorag_research/orm/repository/`)

```python
from autorag_research.orm.repository.base import GenericRepository

class ChunkRepository(GenericRepository[Chunk]):
    """Repository for Chunk operations."""

    def __init__(self, session: Session, model_cls: type[Chunk] | None = None):
        if model_cls is None:
            from autorag_research.orm.schema import Chunk
            model_cls = Chunk
        super().__init__(session, model_cls)

    def get_by_caption_id(self, caption_id: int) -> list[Chunk]:
        """Custom business logic method."""
        return self.session.query(self.model_cls).filter_by(parent_caption=caption_id).all()
```

#### Unit of Work Pattern (`autorag_research/orm/uow/`)

```python
from autorag_research.orm.uow.text_uow import TextOnlyUnitOfWork

with TextOnlyUnitOfWork(session_factory) as uow:
    # Access repositories via properties
    chunks = uow.chunk_repo.get_all()
    uow.chunk_repo.add(new_chunk)
    uow.commit()  # Atomic transaction
```

#### Service Layer (`autorag_research/orm/service/`)

```python
from autorag_research.orm.service.text_ingestion import TextIngestionService

service = TextIngestionService(session_factory)
doc_ids = service.add_documents([{"title": "Doc 1", "author": "Alice"}])
```

### Key Classes

- `GenericRepository[T]`: Base CRUD operations
- `BaseVectorRepository[T]`: Adds vector search (cosine distance, MaxSim)
- `BaseEmbeddingRepository[T]`: Adds embedding-specific queries
- `BaseUnitOfWork`: Transaction management
- `BaseService`: Business logic orchestration

---

## Makefile Commands

```bash
make install      # Install dependencies + pre-commit hooks
make check        # Run linting (ruff), type checking (ty), dependency check (deptry)
make test         # Start PostgreSQL container, run tests, cleanup
make docker-up    # Start PostgreSQL container
make docker-down  # Stop PostgreSQL container
make clean-docker # Remove container and volumes
make build        # Build wheel package
make docs         # Serve documentation locally
make docs-test    # Test documentation build
```

Always use `make check` after implementation to check the typing and linting.
`make test` is for testing, and it is optional when the user asks to run the test.

---

## Testing Guidelines

Tests are located in `tests/` and mirror the package structure.

### Key Testing Rules

1. **Use pytest fixtures** - The `db_session` fixture provides a database session with pre-seeded data
2. **Use existing seed data** - Check `postgresql/db/init/002-seed.sql` for available test data
3. **Clean up after writes** - If you add data for testing, delete it after the test
4. **Single test per function** - Each test function should test one thing
5. **Use `all()` instead of for-loops** in assertions
6. **Async tests** - Use `@pytest.mark.asyncio` decorator
7. **API tests** - Use `@pytest.mark.api` for tests requiring LLM API calls (prefer mocks)
8. **GPU tests** - Use `@pytest.mark.gpu` for tests requiring GPU
9. **No docstrings in tests** - Let the test code speak for itself

### Test Markers

```python
@pytest.mark.gpu    # Requires GPU
@pytest.mark.api    # Requires LLM API call
@pytest.mark.data   # Requires external data download
@pytest.mark.ci_skip  # Skip in CI
```

### Example Test

```python
import pytest
from sqlalchemy.orm import Session
from autorag_research.orm.repository.chunk import ChunkRepository
from autorag_research.orm.schema import Chunk

@pytest.fixture
def chunk_repository(db_session: Session) -> ChunkRepository:
    return ChunkRepository(db_session)

def test_get_by_caption_id(chunk_repository: ChunkRepository, db_session: Session):
    # Use existing seed data (caption id=1 has chunks 1, 2)
    results = chunk_repository.get_by_caption_id(1)

    assert len(results) >= 2
    assert all(c.parent_caption == 1 for c in results)
```

### Running Tests

```bash
# Run all tests (auto-manages PostgreSQL container)
make test

# Run specific test file
uv run pytest tests/autorag_research/orm/repository/test_chunk.py -v

# Run with markers excluded
uv run pytest -m "not gpu and not api and not data"
```

---

## Code Conventions

### Type Hints

- Use Python 3.10+ syntax: `list[str]`, `dict[str, int]`, `str | None`
- **Avoid** `typing.List`, `typing.Dict`, `typing.Optional`

### Linting & Formatting

- **Ruff** for linting and formatting
- Line length: 120 characters
- Pre-commit hooks run automatically

### Ruff Rules

Key enabled rules (see `pyproject.toml`):
- `I` (isort) - import sorting
- `UP` (pyupgrade) - Python version upgrades
- `S` (bandit) - security checks
- `B` (bugbear) - bug detection
- `C4` (comprehensions) - list/dict comprehension improvements

### Exception Handling

Custom exceptions are defined in `autorag_research/exceptions.py`:
- `EnvNotFoundError` - Missing environment variable
- `SessionNotSetError` - Database session not initialized
- `LengthMismatchError` - List length mismatch
- `SchemaNotFoundError` - Schema not found
- `ExecutorError` (base), `PipelineExecutionError`, `MaxRetriesExceededError`
- You can check more exceptions in the file.

---

### Connection String

```python
# psycopg3 (NOT psycopg2!)
postgres_url = f"postgresql+psycopg://{user}:{pwd}@{host}:{port}/{db_name}"

engine = create_engine(
    postgres_url,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)
```

### Vector Search with VectorChord

```python
# Single vector cosine distance
results = chunk_repo.vector_search(query_vector, limit=10)

# Multi-vector MaxSim (late interaction)
results = chunk_repo.maxsim_search(query_vectors, limit=10)
```

---

## Adding New Code

### Adding a New Repository

1. Create file in `autorag_research/orm/repository/`
2. Inherit from `GenericRepository[T]` or `BaseVectorRepository[T]`
3. Add repository property to relevant UoW class
4. Create tests in `tests/autorag_research/orm/repository/`

### Adding a New Pipeline

1. Create config class alongside pipeline. It contains all parameter about the pipeline. Must inherit `BasePipelineConfig` and use `dataclass` decorator.
2. Create pipeline class inheriting from `BaseGenerationPipeline` or `BaseRetrievalPipeline`.
3. Implement `_get_pipeline_config()` and `run()` methods
4. Create tests.

Try to avoid direct use of repository in the pipeline, instead use Service layer.

### Adding a New Metric

1. Create metric function in `autorag_research/evaluation/metrics/`
2. Use `@metric(fields_to_check=[...])` decorator
3. Create config class inheriting from `BaseRetrievalMetricConfig` or `BaseGenerationMetricConfig`
4. Create tests

### Adding a New Service

1. Create file in `autorag_research/orm/service/`
2. Inherit from `BaseService`
3. Implement `_create_uow()` and `_get_schema_classes()`
4. Create tests

### Adding a New Dataset Ingestor.

1. Create file in `autorag_research/data/`
2. Inherit from `BaseDatasetLoader`
3. Be sure to download subset of target dataset and check the structure of dataset. (Be aware finding query, generation_gt, retrieval_gt (qrels), and corpus)
4. Implement data ingestion. The ingestion to the DB and embedding process must be separated.
5. Create tests with 'small subset' of dataset.

---

## Executor Usage

The Executor orchestrates pipeline execution and metric evaluation:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from autorag_research.config import ExecutorConfig
from autorag_research.executor import Executor
from autorag_research.pipelines.retrieval.bm25 import BM25PipelineConfig
from autorag_research.evaluation.metrics.retrieval import RecallConfig, NDCGConfig

engine = create_engine("postgresql+psycopg://...")
session_factory = sessionmaker(bind=engine)

config = ExecutorConfig(
    pipelines=[
        BM25PipelineConfig(
            name="bm25_baseline",
            index_path="/data/index",
            k1=0.9,
            b=0.4,
            top_k=10,
        ),
    ],
    metrics=[
        RecallConfig(),
        NDCGConfig(),
    ],
    max_retries=3,
)

executor = Executor(session_factory, config)
result = executor.run()
```

---

## Important Notes

1. **psycopg3**: This project uses `psycopg` (v3), NOT `psycopg2`. Import as `psycopg`.

2. **VectorChord**: Vector search uses VectorChord extension, not just pgvector. This enables MaxSim operator (`@#`) for multi-vector retrieval.

3. **Schema Factory**: Use `create_schema(dim)` for custom embedding dimensions. The default schema uses 768 dimensions.

4. **Test Isolation**: Tests use transaction rollback for isolation. Don't rely on auto-increment IDs in tests.

5. **Documentation**: Documentation is in `docs/` and can be served with `make docs`. We use Google style annotation and MkDocs.

---

## Useful File Paths

- Database schema: `postgresql/db/init/001-schema.sql`
- Seed data: `postgresql/db/init/002-seed.sql`
- ORM schema factory: `autorag_research/orm/schema_factory.py`
- Base repository: `autorag_research/orm/repository/base.py`
- Test fixtures: `tests/conftest.py`
- Project config: `pyproject.toml`
- DB pattern docs: `ai_instructions/db_pattern.md`
- Test guidelines: `ai_instructions/test_code_generation_instructions.md`
