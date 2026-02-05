import pytest

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

    def test_save_pipeline(self, service, cleanup_pipeline_results):
        pipeline_id = service.save_pipeline(
            name="test_gen_pipeline",
            config={"type": "naive_rag", "model": "test-model"},
        )
        cleanup_pipeline_results.append(pipeline_id)

        assert pipeline_id > 0

        config = service.get_pipeline_config(pipeline_id)
        assert config is not None
        assert config["type"] == "naive_rag"
        assert config["model"] == "test-model"

    def test_run_pipeline(self, service, mock_generate_func, cleanup_pipeline_results, session_factory):
        from autorag_research.orm.repository.query import QueryRepository

        # Count actual queries in database
        with session_factory() as session:
            query_repo = QueryRepository(session)
            query_count = query_repo.count()

        pipeline_id = service.save_pipeline(
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
        from autorag_research.orm.repository.query import QueryRepository

        # Count actual queries in database
        with session_factory() as session:
            query_repo = QueryRepository(session)
            query_count = query_repo.count()

        pipeline_id = service.save_pipeline(
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

    def test_generation_result_without_token_usage(self, service, cleanup_pipeline_results):
        async def generate_func_no_tokens(query_id: int, top_k: int) -> GenerationResult:
            return GenerationResult(text=f"Answer for query: {query_id}", token_usage=None)

        pipeline_id = service.save_pipeline(
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
        from autorag_research.orm.repository.query import QueryRepository

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

        pipeline_id = service.save_pipeline(
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
