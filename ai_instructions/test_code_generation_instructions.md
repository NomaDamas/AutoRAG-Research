# Test Code Generation Instructions

## General Principles

1. **Test-Driven Development (TDD)**: Write tests BEFORE implementation when possible. Tests should be based on the design/specification, not reverse-engineered from the implementation.

2. **Minimal Tests**: Write the minimum number of tests needed to verify correctness. One well-designed test is better than many redundant tests.

3. **Leverage Existing Frameworks**: Use existing test utilities and frameworks. Don't reinvent the wheel.

---

## Pytest Guidelines

1. Try to use pytest fixture when possible. But avoid too much pytest fixture, makes things that can be reusable across multiple tests should be the pytest fixture.
2. We use pytest and its plugins for testing. Avoid to use unittest or other testing framework.
3. We use pytest-asyncio, pytest-env, pytest-mock and other pytest plugins. You can use them when needed. You must add using `uv add <plugin-package> --dev` to the instructions when you use any new pytest plugin.
4. We have PostgreSQL DB with connection in the `conftest.py` file. The name of the session is `db_session`. Try to avoid to create new db session or new database for testing. You can do that when only the user specifically mentions to do it.
5. Try to make a single test function to test a single function. Avoid to make multiple test functions in a single test function. You can create multiple test scenarios in a single function.
6. Always creates test function in the `tests` folder, and make sure the file structure will be identical to the package.
7. The test file should be started with `test_` prefix, and also test function names as well.
8. Try to avoid using for-loop in the assert statement if you can. use `all` and list comprehension instead.
9. In the pytest fixture `db_session`, the data is already added. You can check what was added in the `postgresql/db/init/002-seed.sql` file. Try to use these existing data at all time. Try to avoid to add new data unless you are testing `adding` feature of the db.
10. If you add new data in the db for testing, make sure to remove after the test function is finished.
11. Make sure the setup will be identical to every test sessions.
12. When you need to test async function, make sure to use `pytest.mark.asyncio` decorator.
13. Avoid to use docstring annotation in the test functions. Let the test code speak itself.
14. If the test function uses LLM API call like using LLM or Embedding model, it should be marked as `@pytest.mark.api` or uses mock object. Prefer to use mock object. (Use LlamaIndex MockLLM or MockEmbedding)
15. If the test function uses GPU resource (like local model inference), it should be marked as `@pytest.mark.gpu`.
16. Avoid to use `typing.List`, `typing.Dict` and `typing.Optional`. Use built-in `list`, `dict`, and `|` instead.

---

## Data Ingestor Tests

For data ingestor tests, use the common test framework in `tests/autorag_research/data/ingestor_test_utils.py`.

### Key Principle: Minimal Tests

**DO NOT create tests for:**
- Things already covered by `verify_all()`:
  - Query/chunk/image_chunk count verification
  - ID type validation (string vs bigint)
  - Contents non-empty check
  - Mimetype validation
  - Image data validity (PIL Image.open)
  - Retrieval relation existence
  - Generation GT existence
- `query_limit` / `min_corpus_cnt` parameter variations
- Different limit combinations (both limits, one limit, etc.)
- 1:1 mapping verification for single-relation datasets

**DO create tests for:**
- **Dataset-specific business logic** that transforms data in unique ways
- Examples:
  - ArxivQA: Queries are formatted as `"Given the following query and options...\n\nQuery: ...\n\nOptions: ..."`
  - BEIR HotpotQA: Retrieval GT uses `and_all()` instead of `or_all()`

### Test Structure

```python
# ==================== Unit Tests ====================
# Test helper functions that can be tested without DB/data

class TestHelperFunctions:
    def test_helper_function_basic(self):
        result = _helper_function("input")
        assert result == "expected"


# ==================== Integration Tests ====================
# Use common framework, mark with @pytest.mark.data

CONFIG = IngestorTestConfig(
    expected_query_count=10,
    expected_chunk_count=50,
    chunk_count_is_minimum=True,  # Gold IDs always included
    check_retrieval_relations=True,
    check_generation_gt=True,
    generation_gt_required_for_all=True,  # ALL queries must have GT
    primary_key_type="string",
    db_name="unique_db_name",
)


@pytest.mark.data
class TestIngestorIntegration:
    def test_ingest_subset(self, mock_embedding_model):
        """One test with verify_all() is often sufficient."""
        with create_test_database(CONFIG) as db:
            service = TextDataIngestionService(db.session_factory, schema=db.schema)
            ingestor = MyIngestor(mock_embedding_model)
            ingestor.set_service(service)
            ingestor.ingest(
                query_limit=CONFIG.expected_query_count,
                min_corpus_cnt=CONFIG.expected_chunk_count,
            )
            verifier = IngestorTestVerifier(service, db.schema, CONFIG)
            verifier.verify_all()

    # ONLY add extra tests for dataset-specific business logic
    def test_dataset_specific_format(self, mock_embedding_model):
        """Only if dataset has unique transformations not covered by verify_all()."""
        # ...
```

### Config Options Reference

```python
IngestorTestConfig(
    # Required
    expected_query_count=10,

    # For text datasets
    expected_chunk_count=50,
    # For multi-modal datasets
    expected_image_chunk_count=10,

    # Count verification mode
    chunk_count_is_minimum=False,  # True: >= expected (gold IDs always included)

    # Relation checks
    check_retrieval_relations=True,
    check_generation_gt=False,
    generation_gt_required_for_all=False,  # True: ALL queries must have GT

    # Database
    primary_key_type="string",  # or "bigint"
    db_name="unique_test_db",
)
```

---

## Anti-Patterns to Avoid

1. **Redundant Tests**: Don't test `query_limit=5`, `query_limit=10`, `query_limit=3` separately. One test is enough.

2. **Testing Framework Code**: Don't test that `verify_all()` works - it's already tested. Use it.

3. **Over-Specification**: Don't assert exact content unless it's a specific business requirement.

4. **Implementation-Driven Tests**: Don't look at the implementation and then write tests. Write tests from the specification.

5. **Verbose Tests**: If `verify_all()` covers it, don't add another test for it.
