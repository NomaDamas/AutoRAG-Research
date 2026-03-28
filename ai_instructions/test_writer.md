# Test Writer Shared Instructions

## Role

You write maintainable pytest suites for AutoRAG-Research code while following the project’s layered architecture and test conventions.

## Read First

- `ai_instructions/test_code_generation_instructions.md`
- `ai_instructions/db_pattern.md`
- `ai_instructions/db_schema.md`
- Relevant source file under test
- Matching existing tests in the same area of the repo

## Primary Responsibilities

1. Mirror the source package structure under `tests/`.
2. Use established fixtures and seeded data when appropriate.
3. Follow pytest marker conventions.
4. Avoid redundant or implementation-driven tests.
5. Do not add constructor validation tests.

## General Principles

- One test function should target one behavior.
- Use descriptive `test_<function>_<scenario>_<expected>` naming.
- Prefer existing fixtures from `conftest.py`.
- Clean up data created during the test.
- Use async fixtures for async code.

## Ingestor-Specific Guidance

Use `tests/autorag_research/data/ingestor_test_utils.py` for new data ingestor tests.

- Prefer a single `verify_all()` integration test plus a small number of business-logic tests.
- Do not add tests for count variations or checks already handled by `verify_all()`.
- Use `create_test_database()` instead of `db_session` for ingestor tests.
- Use `FakeEmbeddings` rather than real embedding computation.

## Mocking Rules

- Mock external dependencies at the boundary.
- Do not mock internal services or repositories unless isolation of service logic specifically requires a mocked unit of work.
- Prefer `pytest-mock` utilities where appropriate.

## Test Markers

```python
@pytest.mark.gpu
@pytest.mark.api
@pytest.mark.data
@pytest.mark.asyncio
```

## Anti-Patterns

- Redundant limit-variation tests
- Re-testing helper frameworks already covered elsewhere
- Writing tests from implementation details instead of the spec or observable behavior
- Over-specifying exact content without a business reason

## Output Expectations

- Complete test file with imports
- Appropriate markers
- Clear docstrings when they add value
- Assertions for both expected and edge behavior
