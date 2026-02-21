import pytest

from autorag_research.orm.repository.query import QueryRepository
from autorag_research.orm.service.generation_pipeline import (
    GenerationPipelineService,
    GenerationResult,
)


class TestGenerationPipelineService:
    @pytest.fixture
    def service(self, session_factory):
        return GenerationPipelineService(session_factory)

    @pytest.fixture
    def mock_generate_func(self):
        async def generate_func(query_id: int, top_k: int) -> GenerationResult:
            return GenerationResult(
                text=f"Answer for query: {query_id}",
                token_usage={
                    "prompt_tokens": 50,
                    "completion_tokens": 50,
                    "total_tokens": 100,
                    "embedding_tokens": 0,
                },
                metadata={"retrieved_chunk_ids": [1, 2]},
            )

        return generate_func

    @pytest.fixture
    def cleanup_pipeline_results(self, session_factory):
        created_pipeline_ids = []

        yield created_pipeline_ids

        # Clean up created pipelines
        service = GenerationPipelineService(session_factory)
        for pipeline_id in created_pipeline_ids:
            service.delete_pipeline_results(pipeline_id)

    def test_get_or_create_pipeline(self, service, cleanup_pipeline_results):
        pipeline_id, is_new = service.get_or_create_pipeline(
            name="test_gen_pipeline",
            config={"type": "naive_rag", "model": "test-model"},
        )
        cleanup_pipeline_results.append(pipeline_id)

        assert pipeline_id > 0
        assert is_new is True

        config = service.get_pipeline_config(pipeline_id)
        assert config is not None
        assert config["type"] == "naive_rag"
        assert config["model"] == "test-model"

    def test_run_pipeline(self, service, mock_generate_func, cleanup_pipeline_results, session_factory):

        # Count actual queries in database
        with session_factory() as session:
            query_repo = QueryRepository(session)
            query_count = query_repo.count()

        pipeline_id, _ = service.get_or_create_pipeline(
            name="test_run_gen_pipeline",
            config={"type": "test"},
        )
        cleanup_pipeline_results.append(pipeline_id)

        result = service.run_pipeline(
            generate_func=mock_generate_func,
            pipeline_id=pipeline_id,
            top_k=3,
        )

        assert "pipeline_id" in result
        assert "total_queries" in result
        assert "token_usage" in result
        assert "avg_execution_time_ms" in result
        assert result["pipeline_id"] == pipeline_id
        assert result["total_queries"] == query_count

        # Verify aggregated token usage
        assert result["token_usage"] is not None
        assert result["token_usage"]["total_tokens"] == query_count * 100  # N queries * 100 tokens each
        assert result["token_usage"]["prompt_tokens"] == query_count * 50  # N queries * 50 tokens each
        assert result["token_usage"]["completion_tokens"] == query_count * 50  # N queries * 50 tokens each

    def test_delete_pipeline_results(self, service, mock_generate_func, cleanup_pipeline_results, session_factory):

        # Count actual queries in database
        with session_factory() as session:
            query_repo = QueryRepository(session)
            query_count = query_repo.count()

        pipeline_id, _ = service.get_or_create_pipeline(
            name="test_delete_gen_pipeline",
            config={"type": "test"},
        )
        cleanup_pipeline_results.append(pipeline_id)

        service.run_pipeline(
            generate_func=mock_generate_func,
            pipeline_id=pipeline_id,
            top_k=2,
        )

        # Verify results exist before deletion
        with service._create_uow() as uow:
            results_before = uow.executor_results.get_by_pipeline_id(pipeline_id)
            assert len(results_before) == query_count

        deleted_count = service.delete_pipeline_results(pipeline_id)
        assert deleted_count == query_count

        # Verify results are deleted
        with service._create_uow() as uow:
            results_after = uow.executor_results.get_by_pipeline_id(pipeline_id)
            assert len(results_after) == 0

    def test_get_query_text(self, service):
        query_text = service.get_query_text(3)
        assert query_text == "Three Three"

        query_text = service.get_query_text(1)
        assert query_text == "What is Doc One about?"

    def test_generation_result_without_token_usage(self, service, cleanup_pipeline_results):
        async def generate_func_no_tokens(query_id: int, top_k: int) -> GenerationResult:
            return GenerationResult(text=f"Answer for query: {query_id}", token_usage=None)

        pipeline_id, _ = service.get_or_create_pipeline(
            name="test_no_tokens_pipeline",
            config={"type": "test"},
        )
        cleanup_pipeline_results.append(pipeline_id)

        result = service.run_pipeline(
            generate_func=generate_func_no_tokens,
            pipeline_id=pipeline_id,
            top_k=2,
        )

        assert result["token_usage"] is None

    def test_get_pipeline_config_not_found(self, service):
        config = service.get_pipeline_config(999999)
        assert config is None

    def test_run_pipeline_with_token_usage_jsonb(self, service, cleanup_pipeline_results, session_factory):
        """Test that token_usage JSONB is stored correctly in ExecutorResult."""

        # Count actual queries in database
        with session_factory() as session:
            query_repo = QueryRepository(session)
            query_count = query_repo.count()

        token_usage = {
            "prompt_tokens": 50,
            "completion_tokens": 50,
            "total_tokens": 100,
            "embedding_tokens": 0,
        }

        async def generate_func_with_detail(query_id: int, top_k: int) -> GenerationResult:
            return GenerationResult(
                text=f"Answer for query: {query_id}",
                token_usage=token_usage,
                metadata={"retrieved_chunk_ids": [1, 2]},
            )

        pipeline_id, _ = service.get_or_create_pipeline(
            name="test_token_jsonb_pipeline",
            config={"type": "test"},
        )
        cleanup_pipeline_results.append(pipeline_id)

        result = service.run_pipeline(
            generate_func=generate_func_with_detail,
            pipeline_id=pipeline_id,
            top_k=3,
        )

        # Verify aggregated token usage
        assert result["token_usage"] is not None
        assert result["token_usage"]["total_tokens"] == query_count * 100  # N queries * 100 tokens each

        # Verify token_usage is stored in ExecutorResult as JSONB
        with service._create_uow() as uow:
            executor_results = uow.executor_results.get_by_pipeline_id(pipeline_id)
            assert len(executor_results) == query_count

            for exec_result in executor_results:
                assert exec_result.token_usage is not None
                assert exec_result.token_usage["prompt_tokens"] == 50
                assert exec_result.token_usage["completion_tokens"] == 50
                assert exec_result.token_usage["total_tokens"] == 100
                assert exec_result.token_usage["embedding_tokens"] == 0


class TestGetOrCreateGenerationPipeline:
    """Tests for GenerationPipelineService.get_or_create_pipeline method."""

    @pytest.fixture
    def service(self, session_factory):
        return GenerationPipelineService(session_factory)

    @pytest.fixture
    def unique_name(self):
        """Generate a unique pipeline name to avoid collisions with stale test data."""
        import uuid

        return f"test_gen_gocp_{uuid.uuid4().hex[:8]}"

    def test_get_or_create_pipeline_new(self, service, unique_name):
        """When no pipeline with the name exists, a new one is created."""
        pipeline_id, is_new = service.get_or_create_pipeline(
            name=unique_name,
            config={"type": "naive_rag", "model": "gpt-4"},
        )

        assert is_new is True
        assert pipeline_id is not None

        config = service.get_pipeline_config(pipeline_id)
        assert config == {"type": "naive_rag", "model": "gpt-4"}

    def test_get_or_create_pipeline_existing(self, service, unique_name):
        """When a pipeline with the name exists, its ID is returned."""
        pipeline_id, is_new = service.get_or_create_pipeline(
            name=unique_name,
            config={"type": "naive_rag", "model": "gpt-4"},
        )
        assert is_new is True

        # Second call should find the existing pipeline
        pipeline_id2, is_new2 = service.get_or_create_pipeline(
            name=unique_name,
            config={"type": "naive_rag", "model": "gpt-4"},
        )

        assert is_new2 is False
        assert pipeline_id2 == pipeline_id

    def test_get_or_create_pipeline_config_mismatch(self, service, unique_name, caplog):
        """When pipeline exists with different config, logs a warning and reuses it."""
        import logging

        pipeline_id, _ = service.get_or_create_pipeline(
            name=unique_name,
            config={"type": "naive_rag", "model": "gpt-4"},
        )

        with caplog.at_level(logging.WARNING, logger="AutoRAG-Research"):
            pipeline_id2, is_new = service.get_or_create_pipeline(
                name=unique_name,
                config={"type": "naive_rag", "model": "gpt-3.5"},
            )

        assert is_new is False
        assert pipeline_id2 == pipeline_id
        assert "different config" in caplog.text


class TestGenerationPipelineResume:
    """Tests for query skip logic in generation run_pipeline (resume support)."""

    @pytest.fixture
    def service(self, session_factory):
        return GenerationPipelineService(session_factory)

    @pytest.fixture
    def cleanup_pipeline_results(self, session_factory):
        created_pipeline_ids = []

        yield created_pipeline_ids

        service = GenerationPipelineService(session_factory)
        for pipeline_id in created_pipeline_ids:
            service.delete_pipeline_results(pipeline_id)

    def test_run_pipeline_skips_completed_queries(self, service, cleanup_pipeline_results, session_factory):
        """Pre-insert results for some queries, verify generate_func not called for them."""
        with session_factory() as session:
            query_repo = QueryRepository(session)
            query_count = query_repo.count()

        pipeline_id, _ = service.get_or_create_pipeline(
            name="test_skip_completed_gen",
            config={"type": "test"},
        )
        cleanup_pipeline_results.append(pipeline_id)

        # Pre-insert an executor result for the first query
        with service._create_uow() as uow:
            queries = uow.queries.get_all(limit=1, offset=0)
            first_query_id = queries[0].id
            executor_result_class = service._get_schema_classes()["ExecutorResult"]
            uow.executor_results.add(
                executor_result_class(
                    query_id=first_query_id,
                    pipeline_id=pipeline_id,
                    generation_result="Pre-existing answer",
                    token_usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                    execution_time=100,
                )
            )
            uow.commit()

        call_log: list[int | str] = []

        async def tracked_generate(query_id: int | str, top_k: int) -> GenerationResult:
            call_log.append(query_id)
            return GenerationResult(text=f"Answer for {query_id}", token_usage=None)

        result = service.run_pipeline(
            generate_func=tracked_generate,
            pipeline_id=pipeline_id,
            top_k=1,
        )

        assert first_query_id not in call_log
        assert result["total_queries"] == query_count - 1

    def test_run_pipeline_all_completed(self, service, cleanup_pipeline_results, session_factory):
        """When all queries have results, processes 0 queries."""
        pipeline_id, _ = service.get_or_create_pipeline(
            name="test_all_completed_gen",
            config={"type": "test"},
        )
        cleanup_pipeline_results.append(pipeline_id)

        # Pre-insert executor results for ALL queries
        with service._create_uow() as uow:
            queries = uow.queries.get_all(limit=1000, offset=0)
            executor_result_class = service._get_schema_classes()["ExecutorResult"]
            for q in queries:
                uow.executor_results.add(
                    executor_result_class(
                        query_id=q.id,
                        pipeline_id=pipeline_id,
                        generation_result=f"Answer for {q.id}",
                        token_usage=None,
                        execution_time=50,
                    )
                )
            uow.commit()

        call_log: list[int | str] = []

        async def tracked_generate(query_id: int | str, top_k: int) -> GenerationResult:
            call_log.append(query_id)
            return GenerationResult(text="should not be called", token_usage=None)

        result = service.run_pipeline(
            generate_func=tracked_generate,
            pipeline_id=pipeline_id,
            top_k=1,
        )

        assert result["total_queries"] == 0
        assert len(call_log) == 0

    def test_run_pipeline_backward_compat(self, service, cleanup_pipeline_results, session_factory):
        """New pipeline with no pre-existing results processes all queries."""
        with session_factory() as session:
            query_repo = QueryRepository(session)
            query_count = query_repo.count()

        async def generate_func(query_id: int | str, top_k: int) -> GenerationResult:
            return GenerationResult(text=f"Answer for {query_id}", token_usage=None)

        pipeline_id, _ = service.get_or_create_pipeline(
            name="test_backward_compat_gen",
            config={"type": "test"},
        )
        cleanup_pipeline_results.append(pipeline_id)

        result = service.run_pipeline(
            generate_func=generate_func,
            pipeline_id=pipeline_id,
            top_k=1,
        )

        assert result["total_queries"] == query_count
