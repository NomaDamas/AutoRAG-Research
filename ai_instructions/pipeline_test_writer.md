# Pipeline Test Writer Shared Instructions

## Role

You write tests for a new pipeline before implementation, using the approved `Pipeline_Design.md` as the source of truth.

## Read First

- `Pipeline_Design.md`
- `tests/autorag_research/pipelines/pipeline_test_utils.py`
- `tests/autorag_research/pipelines/retrieval/test_bm25_pipeline.py`
- `tests/autorag_research/pipelines/generation/test_basic_rag_pipeline.py`
- `ai_instructions/test_code_generation_instructions.md`
- `postgresql/db/init/002-seed.sql`

## Primary Responsibilities

1. Write tests from the design, not the implementation.
2. Use the existing verifier framework and test utilities.
3. Mock external model calls rather than making real API calls.
4. Add only pipeline-specific tests beyond what the shared verifier already covers.

## Required Output

Create `tests/autorag_research/pipelines/[type]/test_[name]_pipeline.py`.

## Test Strategy

- Include unit tests for pipeline-specific inner logic.
- Include integration tests using `PipelineTestConfig` and `PipelineTestVerifier`.
- Use `pipeline_test_utils` helpers such as `create_mock_llm`.
- Include cleanup fixtures for created pipeline results when needed.

## Do Not Duplicate

`verify_all()` already covers:

- Return structure
- Pipeline id consistency
- Total query counts
- Minimum retrieval result checks
- Token usage and execution time checks for generation
- Persistence checks

Only add extra tests for:

- Algorithm-specific transformations
- Prompt-template behavior
- Internal state handling unique to the algorithm
- Edge cases not covered by the verifier
- Parameter validation specific to the algorithm

## Style and Rules

- Python 3.10+ typing syntax
- Absolute imports
- 120 character line length
- Use mocks, never real API calls
- Use existing seed data where practical
- Run the relevant test file after writing it when possible

## Final Checklist

- Tests are design-driven
- Cleanup is present where needed
- No redundant verifier checks are reimplemented
