# Workflow: Implementing a New Pipeline

This document defines the workflow for implementing a new Pipeline (Retrieval or Generation) in **AutoRAG-Research** using specialized sub-agents.

**IMPORTANT:** Intermediate artifacts (`Pipeline_Analysis.json`, `Pipeline_Design.md`) are for development use only. **Do not commit or push these files to the git repository.**

## Agents

1. **Paper Analyst:** Analyzes research papers (PDF, arxiv links) to extract algorithm details.
2. **Architecture Mapper:** Maps the algorithm to existing patterns (base classes, services, UoW).
3. **Test Writer:** Creates tests BEFORE implementation (TDD approach).
4. **Implementation Specialist:** Writes production code to pass the tests.
5. **Validation Agent:** Runs tests, type checks, and linting.

**Note:** Code quality checks (`make check`) run automatically via hooks after file edits.

## Workflow Steps (Test-Driven Development)

### Phase 1: Investigation (Paper Analysis)

* **Agent:** Paper Analyst
* **Input:** Paper (PDF, arxiv link, or other URL) from Issue.
* **Output:** `Pipeline_Analysis.json` (Local only. Do not commit).

**Output Content:**
```json
{
  "algorithm_name": "HyDE",
  "pipeline_type": "retrieval",
  "core_steps": [
    "1. Generate hypothetical document using LLM",
    "2. Embed hypothetical document",
    "3. Retrieve real documents by similarity"
  ],
  "parameters": {
    "llm": "LLM instance for hypothesis generation",
    "embedding_model": "Embedding model for encoding",
    "num_hypotheses": "Number of hypothetical documents to generate"
  },
  "dependencies": ["llama_index.llms", "llama_index.embeddings"],
  "method_signatures": {
    "_get_retrieval_func": "Returns retrieval function",
    "_get_pipeline_config": "Returns config dict with type and parameters"
  }
}
```

### Phase 2: Design (Architecture Mapping)

* **Agent:** Architecture Mapper
* **Input:** `Pipeline_Analysis.json`, existing patterns.
* **Required Reading:** Base classes, config classes, reference implementations.
* **Human Checkpoint:** MANDATORY approval before proceeding.
* **Output:** `Pipeline_Design.md` (Local only. Do not commit).

**Files to Read:**

| Purpose | File Path |
|---------|-----------|
| Retrieval base class | `autorag_research/pipelines/retrieval/base.py` |
| Generation base class | `autorag_research/pipelines/generation/base.py` |
| Config base classes | `autorag_research/config.py` |
| Retrieval example | `autorag_research/pipelines/retrieval/bm25.py` |
| Generation example | `autorag_research/pipelines/generation/basic_rag.py` |

**Output Content:**

```markdown
# Pipeline Design: [Algorithm Name]

## Pipeline Type
Retrieval / Generation

## Base Class
`BaseRetrievalPipeline` / `BaseGenerationPipeline`

## Constructor Parameters
- `session_factory`: SQLAlchemy sessionmaker
- `name`: Pipeline name
- `param1`: Description (type, default)
- `param2`: Description (type, default)
- `schema`: Optional dynamic schema

## Abstract Methods to Implement

### _get_pipeline_config()
```python
def _get_pipeline_config(self) -> dict[str, Any]:
    return {
        "type": "algorithm_name",
        "param1": self.param1,
        "param2": self.param2,
    }
```

### _get_retrieval_func() / _generate()
[Detailed implementation plan]

## Config Dataclass

```python
@dataclass(kw_only=True)
class AlgorithmPipelineConfig(BaseRetrievalPipelineConfig):
    param1: str
    param2: float = 0.5

    def get_pipeline_class(self) -> type["AlgorithmPipeline"]:
        return AlgorithmPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        return {"param1": self.param1, "param2": self.param2}
```
```

### Phase 3: Testing (TDD)

* **Agent:** Test Writer
* **Input:** `Pipeline_Design.md`, Dataset characteristics from Phase 1-2.
* **Human Checkpoint:** Review test plan before writing.
* **Output:** Test file (Commit this file).
* **Location:** `tests/autorag_research/pipelines/[type]/test_[name]_pipeline.py`

**Key Principle:** Write tests FIRST based on the design document, NOT after seeing the implementation.

#### Test Structure

**1. Unit Tests** - Test inner logic directly:

```python
class TestAlgorithmPipelineUnit:
    def test_pipeline_creation(self, pipeline):
        """Test that pipeline is created correctly."""
        assert pipeline.pipeline_id > 0
        assert pipeline.param1 == expected_value

    def test_pipeline_config(self, pipeline):
        """Test that pipeline config is stored correctly."""
        config = pipeline._get_pipeline_config()
        assert config["type"] == "algorithm_name"
        assert config["param1"] == expected_value

    def test_single_query(self, pipeline):
        """Test single query processing."""
        # For retrieval: pipeline.retrieve(query, top_k)
        # For generation: pipeline._generate(query, top_k)
        pass
```

**2. Integration Tests** - Use `PipelineTestConfig` and `PipelineTestVerifier`:

```python
def test_run_pipeline(self, pipeline, session_factory):
    """Test running the full pipeline with verification."""
    result = pipeline.run(top_k=3, batch_size=10)

    config = PipelineTestConfig(
        pipeline_type="retrieval",  # or "generation"
        expected_total_queries=5,   # Seed data default
        expected_min_results=0,     # For retrieval only
        check_token_usage=True,     # For generation only
        check_execution_time=True,  # For generation only
        check_persistence=True,
    )
    verifier = PipelineTestVerifier(result, pipeline.pipeline_id, session_factory, config)
    verifier.verify_all()
```

#### What verify_all() Already Checks (DO NOT Duplicate)

| Check | Description |
|-------|-------------|
| Return structure | Dict with required keys (pipeline_id, total_queries, etc.) |
| pipeline_id | Matches expected pipeline ID |
| total_queries | Matches expected count |
| total_results | >= expected minimum (retrieval) |
| token_usage | Dict with prompt/completion/total/embedding tokens (generation) |
| avg_execution_time_ms | Valid positive number (generation) |
| Persistence | Database records exist (ChunkRetrievedResult or ExecutorResult) |

#### Add Extra Tests ONLY For

- **Pipeline-specific transformations** (e.g., HyDE's hypothesis generation)
- **Custom prompt template usage** (verify template is applied correctly)
- **Edge cases specific to algorithm** (e.g., empty retrieval results handling)
- **Internal state management** (e.g., token counter reset between generations)
- **Parameter validation** (e.g., k1/b values for BM25)

Example:
```python
def test_custom_prompt_template(self, session_factory, mock_llm, retrieval_pipeline):
    """Test pipeline with custom prompt template (business logic specific to this pipeline)."""
    custom_template = "Documents:\n{context}\n\nQuery: {query}\n\nResponse:"

    pipeline = BasicRAGPipeline(
        session_factory=session_factory,
        name="test_custom_template",
        llm=mock_llm,
        retrieval_pipeline=retrieval_pipeline,
        prompt_template=custom_template,
    )

    _ = pipeline._generate("Test query", top_k=2)

    # Verify the custom template was used
    call_args = mock_llm.complete.call_args
    prompt = call_args[0][0]
    assert "Documents:" in prompt
    assert "Response:" in prompt
```

### Phase 4: Implementation

* **Agent:** Implementation Specialist
* **Input:** `Pipeline_Design.md`, Test file from Phase 3.
* **Output:** Pipeline class file (Commit this file).
* **Location:** `autorag_research/pipelines/[type]/[name].py`

**Key Principle:** Implementation should pass the tests written in Phase 3.

**Critical Pattern - Store Parameters BEFORE super().__init__():**

```python
class AlgorithmPipeline(BaseRetrievalPipeline):
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        param1: str,
        param2: float = 0.5,
        schema: Any | None = None,
    ):
        # IMPORTANT: Store parameters BEFORE calling super().__init__
        # because _get_pipeline_config() is called during base init
        self.param1 = param1
        self.param2 = param2

        super().__init__(session_factory, name, schema)

    def _get_pipeline_config(self) -> dict[str, Any]:
        return {
            "type": "algorithm_name",
            "param1": self.param1,
            "param2": self.param2,
        }

    def _get_retrieval_func(self) -> Any:
        # Implementation here
        pass
```

### Phase 5: Validation

* **Agent:** Validation Agent
* **Input:** Implementation from Phase 4.
* **Human Checkpoint:** Final review before commit.

**Steps:**
1. Run tests: `uv run pytest tests/autorag_research/pipelines/[type]/test_[name]_pipeline.py -v`
2. Run checks: `make check`
3. Verify all tests pass
4. Human review of implementation quality

---

## Common Test Framework

All pipeline tests use the common utilities in `tests/autorag_research/pipelines/pipeline_test_utils.py`.

### PipelineTestConfig Options

```python
@dataclass
class PipelineTestConfig:
    # Pipeline type
    pipeline_type: Literal["retrieval", "generation"]

    # Common
    expected_total_queries: int = 5  # Seed data default

    # Retrieval pipeline specific
    expected_min_results: int | None = None

    # Generation pipeline specific
    check_token_usage: bool = False
    check_execution_time: bool = False
    expected_token_usage_keys: list[str] = field(
        default_factory=lambda: ["prompt_tokens", "completion_tokens", "total_tokens", "embedding_tokens"]
    )

    # Persistence check
    check_persistence: bool = True

    # Schema for dynamic schema support
    schema: Any | None = None
```

### Mock Utilities

```python
from tests.autorag_research.pipelines.pipeline_test_utils import create_mock_llm

# Create mock LLM for generation pipeline tests
mock_llm = create_mock_llm(
    response_text="This is a generated answer.",
    token_usage={
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
    },
)
```

### Cleanup Fixture Pattern

```python
@pytest.fixture
def cleanup_pipeline_results(self, session_factory):
    """Cleanup fixture that deletes pipeline results after test."""
    created_pipeline_ids = []

    yield created_pipeline_ids

    session = session_factory()
    try:
        # For retrieval pipelines
        result_repo = ChunkRetrievedResultRepository(session)
        # For generation pipelines, also add:
        # executor_repo = ExecutorResultRepository(session)
        for pipeline_id in created_pipeline_ids:
            result_repo.delete_by_pipeline(pipeline_id)
        session.commit()
    finally:
        session.close()
```

---

## Key Principles

1. **TDD:** Write tests FIRST based on design, not after implementation
2. **Minimal Tests:** Use `verify_all()` for standard checks, only add tests for pipeline-specific logic
3. **Store Parameters Early:** Set instance variables BEFORE `super().__init__()` call
4. **Use MockLLM:** Never make real API calls in tests
5. **Seed Data:** Use existing seed data (5 queries, 6 chunks) for integration tests
6. **Cleanup:** Always clean up created pipeline results in tests

---

## Definition of Done

* [ ] `Pipeline_Analysis.json` generated (Local).
* [ ] `Pipeline_Design.md` approved by human (Local).
* [ ] **Tests written FIRST** based on design document.
* [ ] Pipeline class implemented.
* [ ] Config dataclass implemented.
* [ ] `make check` passes.
* [ ] All tests pass.
* [ ] Intermediate files removed or excluded from git.
* [ ] PR ready with branch `Feature/#[IssueID]`.

---

## Retrieval vs Generation Comparison

| Aspect | Retrieval Pipeline | Generation Pipeline |
|--------|-------------------|---------------------|
| Base Class | `BaseRetrievalPipeline` | `BaseGenerationPipeline` |
| Core Method | `_get_retrieval_func()` | `_generate()` |
| Return Type | `dict[str, Any]` with `total_results` | `dict[str, Any]` with `token_usage`, `avg_execution_time_ms` |
| Service | `RetrievalPipelineService` | `GenerationPipelineService` |
| Result Table | `ChunkRetrievedResult` | `ExecutorResult` |
| Config Base | `BaseRetrievalPipelineConfig` | `BaseGenerationPipelineConfig` |
| Single Query | `.retrieve(query, top_k)` | `._generate(query, top_k)` |
| Dependencies | May need index paths, models | Requires LLM + retrieval pipeline |

---

## Critical Files Reference

| Purpose | File Path |
|---------|-----------|
| Retrieval base class | `autorag_research/pipelines/retrieval/base.py` |
| Generation base class | `autorag_research/pipelines/generation/base.py` |
| Config base classes | `autorag_research/config.py` |
| Test utilities | `tests/autorag_research/pipelines/pipeline_test_utils.py` |
| Retrieval example | `autorag_research/pipelines/retrieval/bm25.py` |
| Generation example | `autorag_research/pipelines/generation/basic_rag.py` |
| Retrieval test example | `tests/autorag_research/pipelines/retrieval/test_bm25_pipeline.py` |
| Generation test example | `tests/autorag_research/pipelines/generation/test_basic_rag_pipeline.py` |
