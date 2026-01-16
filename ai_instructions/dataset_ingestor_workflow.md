# Workflow: Adding a New Dataset Ingestor

This document defines the workflow for implementing a new Dataset Ingestor in **AutoRAG-Research** using specialized sub-agents.

**IMPORTANT:** Intermediate artifacts (JSON analysis, Markdown strategy) are for development use only. **Do not commit or push these files to the git repository.**

## Agents

1. **Dataset Inspector:** Analyzes raw external data structure.
2. **Schema Architect:** Maps raw data to internal DB schema.
3. **Implementation Specialist:** Writes production code.
4. **Test Writer:** Creates integration tests using the common test framework.

During implementation, agents will run `make check` (ruff linting, ty type checking, deptry) to verify code quality and fix any issues.

## Workflow Steps

### Phase 1: Investigation

* **Agent:** Dataset Inspector
* **Input:** Dataset Name/Link from Issue.
* **Output:** `Source_Data_Profile.json` (Local only. Do not commit).

### Phase 2: Design

* **Agent:** Schema Architect
* **Input:** `Source_Data_Profile.json`, `ai_instructions/db_schema.md`.
* **Output:** `Mapping_Strategy.md` (Local only. Do not commit).

### Phase 3: Implementation

* **Agent:** Implementation Specialist
* **Input:** `Mapping_Strategy.md`.
* **Output:** `autorag_research/data/[dataset_name].py` (Commit this file).

### Phase 4: Testing

* **Agent:** Test Writer
* **Input:** Ingestor Source Code, Dataset characteristics.
* **Output:** `tests/autorag_research/data/test_[dataset_name].py` (Commit this file).

#### Common Test Framework

All ingestor tests use the common test utilities in `tests/autorag_research/data/ingestor_test_utils.py`.

**Key Components:**

1. **`IngestorTestConfig`** - Configuration dataclass for test parameters:
   - `expected_query_count`: Number of queries to ingest and verify
   - `expected_chunk_count`: Number of text chunks (for text datasets)
   - `expected_image_chunk_count`: Number of image chunks (for multi-modal datasets)
   - `check_retrieval_relations`: Verify every query has retrieval relations
   - `check_generation_gt`: Verify queries have generation ground truth
   - `primary_key_type`: "string" or "bigint" based on dataset
   - `db_name`: Unique database name for test isolation

2. **`create_test_database(config)`** - Context manager that:
   - Creates isolated PostgreSQL test database
   - Installs vector extensions
   - Creates schema with correct primary key type
   - Cleans up database after test completion

3. **`IngestorTestVerifier`** - Verifies ingested data:
   - Count verification (queries, chunks, etc.)
   - Random sample format validation
   - Retrieval relation verification
   - Content hash verification (optional)
   - Logs sample content for CI inspection

**Test Template:**

```python
from tests.autorag_research.data.ingestor_test_utils import (
    IngestorTestConfig,
    IngestorTestVerifier,
    create_test_database,
)

CONFIG = IngestorTestConfig(
    expected_query_count=10,
    expected_chunk_count=50,
    check_retrieval_relations=True,
    check_generation_gt=True,  # Set based on dataset
    primary_key_type="string",  # or "bigint"
    db_name="my_dataset_test",
)

@pytest.mark.data
def test_my_ingestor_integration(mock_embedding_model):
    with create_test_database(CONFIG) as db:
        service = TextDataIngestionService(db.session_factory, schema=db.schema)

        ingestor = MyIngestor(mock_embedding_model)
        ingestor.set_service(service)
        ingestor.ingest(
            query_limit=CONFIG.expected_query_count,
            corpus_limit=CONFIG.expected_chunk_count,
        )

        verifier = IngestorTestVerifier(service, db.schema, CONFIG)
        verifier.verify_all()
```

**Guidelines:**

1. Use `MockEmbedding` for all tests (no actual embedding computation)
2. Use `query_limit` and `corpus_limit` for fast CI tests (default: 10 queries, 50 corpus)
3. Keep unit tests for helper functions separate from integration tests
4. Mark integration tests with `@pytest.mark.data`
5. Each test should use a unique `db_name` to enable parallel test execution

## Definition of Done

* [ ] `Source_Data_Profile.json` generated (Local).
* [ ] `Mapping_Strategy.md` generated (Local).
* [ ] Ingestor class implemented in `autorag_research/data`.
* [ ] Integration tests implemented using common test framework.
* [ ] Static analysis (Lint/Type) passed.
* [ ] Integration tests pass against real data subsets.
* [ ] Intermediate files removed or excluded from git.
* [ ] PR ready with branch `Feature/#[IssueID]`.
