# Workflow: Adding a New Dataset Ingestor

This document defines the workflow for implementing a new Dataset Ingestor in **AutoRAG-Research** using specialized sub-agents.

**IMPORTANT:** Intermediate artifacts (JSON analysis, Markdown strategy) are for development use only. **Do not commit or push these files to the git repository.**

## Agents

1. **Dataset Inspector:** Analyzes raw external data structure.
2. **Schema Architect:** Maps raw data to internal DB schema.
3. **Test Writer:** Creates minimal integration tests BEFORE implementation.
4. **Implementation Specialist:** Writes production code to pass the tests.

## Skills

- **`/design-ingestor`:** Orchestrates Phase 2 with **mandatory human review**. Use this skill instead of calling schema-architect directly.

**Note:** Code quality checks (`make check`) run automatically via hooks after file edits.

## Workflow Steps (Test-Driven Development)

### Phase 1: Investigation

* **Agent:** Dataset Inspector
* **Input:** Dataset Name/Link from Issue.
* **Output:** `Source_Data_Profile.json` (Local only. Do not commit).

### Phase 2: Design (with Human Review)

* **Skill:** `/design-ingestor` (MANDATORY - do not call schema-architect directly)
* **Input:** `Source_Data_Profile.json`, `ai_instructions/db_schema.md`.
* **Output:** `Mapping_Strategy.md` (Local only. Do not commit).

**Human-in-the-Loop:** The `/design-ingestor` skill enforces a mandatory review cycle:
1. Schema Architect generates the mapping strategy
2. Strategy summary is presented to user
3. User must explicitly **Approve**, **Request Changes**, or **Reject**
4. If changes requested → revise and re-present until approved
5. **Phase 3 cannot begin without explicit approval**

### Phase 3: Testing (BEFORE Implementation)

* **Agent:** Test Writer
* **Input:** `Mapping_Strategy.md`, Dataset characteristics from Phase 1-2.
* **Output:** `tests/autorag_research/data/test_[dataset_name].py` (Commit this file).

**Key Principle:** Write tests FIRST based on the design document, NOT after seeing the implementation.

### Phase 4: Implementation

* **Agent:** Implementation Specialist
* **Input:** `Mapping_Strategy.md`, Test file from Phase 3.
* **Output:** `autorag_research/data/[dataset_name].py` (Commit this file).

**Key Principle:** Implementation should pass the tests written in Phase 3.

---

## Common Test Framework

All ingestor tests use the common test utilities in `tests/autorag_research/data/ingestor_test_utils.py`.

### IngestorTestConfig Options

```python
@dataclass
class IngestorTestConfig:
    # Required counts
    expected_query_count: int
    expected_chunk_count: int | None = None        # For text datasets
    expected_image_chunk_count: int | None = None  # For multi-modal datasets

    # Count verification mode
    chunk_count_is_minimum: bool = False  # True: >= expected, False: == expected

    # Relation checks
    check_retrieval_relations: bool = True
    check_generation_gt: bool = False
    generation_gt_required_for_all: bool = False  # True: ALL queries must have GT

    # Database settings
    primary_key_type: Literal["bigint", "string"] = "string"
    db_name: str = "ingestor_test_db"
```

### What `verify_all()` Already Checks

The `IngestorTestVerifier.verify_all()` method automatically verifies:

1. **Count verification**: Query count, chunk count, image chunk count
2. **Format validation**: Random sample of records for correct typing/format
3. **Image validation**: PIL Image.open() to verify valid image data
4. **Retrieval relations**: Every query has at least one relation
5. **Generation GT**: Queries have valid generation ground truth (if enabled)
6. **Sample logging**: Logs content samples for CI inspection

### Minimal Test Template

```python
import pytest
from llama_index.core import MockEmbedding

from autorag_research.data.my_dataset import MyDatasetIngestor
from autorag_research.orm.service.text_ingestion import TextDataIngestionService
from tests.autorag_research.data.ingestor_test_utils import (
    IngestorTestConfig,
    IngestorTestVerifier,
    create_test_database,
)

EMBEDDING_DIM = 768


@pytest.fixture
def mock_embedding_model():
    return MockEmbedding(EMBEDDING_DIM)


# ==================== Integration Tests ====================

CONFIG = IngestorTestConfig(
    expected_query_count=10,
    expected_chunk_count=50,
    chunk_count_is_minimum=True,  # For datasets where gold IDs are always included
    check_retrieval_relations=True,
    check_generation_gt=True,
    generation_gt_required_for_all=True,  # If ALL queries must have answers
    primary_key_type="string",
    db_name="my_dataset_test",
)


@pytest.mark.data
class TestMyDatasetIngestorIntegration:
    def test_ingest_subset(self, mock_embedding_model):
        """Basic integration test - verify_all() handles all standard checks."""
        with create_test_database(CONFIG) as db:
            service = TextDataIngestionService(db.session_factory, schema=db.schema)

            ingestor = MyDatasetIngestor(mock_embedding_model)
            ingestor.set_service(service)
            ingestor.ingest(
                query_limit=CONFIG.expected_query_count,
                min_corpus_cnt=CONFIG.expected_chunk_count,
            )

            verifier = IngestorTestVerifier(service, db.schema, CONFIG)
            verifier.verify_all()
```

### When to Add Extra Tests

**DO NOT add extra tests for:**
- Query/corpus limit functionality (covered by verify_all count checks)
- Image mimetype validation (covered by verify_all image format check)
- Retrieval relation existence (covered by verify_all)
- Generation GT existence (covered by verify_all with `generation_gt_required_for_all`)

**DO add extra tests for:**
- **Dataset-specific business logic** that verify_all() cannot check
- Example: ArxivQA formats queries as `"Given the following query and options...\n\nQuery: ...\n\nOptions: ..."`

```python
def test_query_contents_format(self, mock_embedding_model):
    """Test dataset-specific query format (ArxivQA-specific business logic)."""
    with create_test_database(config) as db:
        # ... setup ...

        with service._create_uow() as uow:
            queries = uow.queries.get_all(limit=10)
            for query in queries:
                # ArxivQA-specific format validation
                assert "Query:" in query.contents
                assert "Options:" in query.contents
```

### Guidelines

1. **Minimal tests**: One integration test using `verify_all()` is often sufficient
2. **No redundant tests**: Don't test what verify_all() already checks
3. **Business logic only**: Only add extra tests for dataset-specific transformations
4. **Use MockEmbedding**: No actual embedding computation
5. **Small subsets**: Default 10 queries, 50 corpus for fast CI
6. **Unique db_name**: Each test uses isolated database

---

## Decorator Registration (Required)

All Ingestors must be registered with the `@register_ingestor` decorator.
CLI options are automatically extracted from the `__init__` signature.

### Example

```python
from typing import Literal
from llama_index.core.base.embeddings.base import BaseEmbedding
from autorag_research.data.registry import register_ingestor
from autorag_research.data.base import TextEmbeddingDataIngestor

# Define available datasets as Literal type
MY_DATASETS = Literal["dataset_a", "dataset_b", "dataset_c"]

@register_ingestor(
    name="my_dataset",
    description="My dataset ingestor",
)
class MyDatasetIngestor(TextEmbeddingDataIngestor):
    def __init__(
        self,
        embedding_model: BaseEmbedding,  # Skipped (known dependency)
        config_name: MY_DATASETS,         # -> --config-name, choices=[...], required
        batch_size: int = 100,            # -> --batch-size, default=100
    ):
        super().__init__(embedding_model)
        self.config_name = config_name
        self.batch_size = batch_size
```

### Auto-Inference Rules

| `__init__` Parameter | CLI Option |
|-------------------|---------|
| `embedding_model: BaseEmbedding` | Skipped |
| `name: Literal["a", "b"]` | `--name`, choices=["a", "b"], required |
| `name: Literal["a", "b"] = "a"` | `--name`, choices=["a", "b"], default="a" |
| `name: str` | `--name`, required |
| `count: int = 10` | `--count`, type=int, default=10 |
| `items: list[str]` | `--items`, comma-separated, is_list=True |

### CLI Verification

```bash
# Check registered ingestors
autorag-research list ingestors

# Check specific ingestor options
autorag-research ingest my_dataset --help
```

---

## Definition of Done

* [ ] `Source_Data_Profile.json` generated (Local).
* [ ] `Mapping_Strategy.md` generated (Local).
* [ ] **Human review completed** - Strategy explicitly approved via `/design-ingestor`.
* [ ] **Tests written FIRST** based on design document.
* [ ] Ingestor class implemented to pass the tests.
* [ ] **`@register_ingestor` decorator is applied**.
* [ ] **Define choices by `Literal` type**.
* [ ] **New dataset ingestor is shown in the 'autorag-research list ingestors' cli command**.
* [ ] Static analysis (Lint/Type) passed.
* [ ] Integration tests pass against real data subsets.
* [ ] Intermediate files removed or excluded from git.
* [ ] PR ready with branch `Feature/#[IssueID]`.
