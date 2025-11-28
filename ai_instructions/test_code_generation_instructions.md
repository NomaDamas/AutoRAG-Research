1. Try to use pytest fixture when possible. But avoid too much pytest fixture, makes things that can be reusable across multiple tests should be the pytest fixture.
2. We use pytest and its plugins for testing. Avoid to use unittest or other testing framework.
3. We use pytest-asyncio, pytest-env, pytest-mock and other pytest plugins. You can use them when needed. You must add using `uv add <plugin-package> --dev` to the instructions when you use any new pytest plugin.
4. We have PostgreSQL DB with connection in the `conftest.py` file. The name of the session is `db_session`. Try to avoid to create new db session or new database for testing. You can do that when only the user specifially mentions to do it.
5. Try to make a single test function to test a single function. Avoid to make mulitple test functions in a single test function. You can create multiple test scenarios in a single function.
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
