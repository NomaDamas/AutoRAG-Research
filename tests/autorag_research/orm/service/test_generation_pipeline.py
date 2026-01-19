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
        def generate_func(query: str, top_k: int) -> GenerationResult:
            return GenerationResult(
                text=f"Answer for: {query}",
                token_usage=100,
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

    def test_run_pipeline(self, service, mock_generate_func, cleanup_pipeline_results):
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
        assert "total_tokens" in result
        assert "avg_execution_time_ms" in result
        assert result["pipeline_id"] == pipeline_id
        assert result["total_queries"] == 5  # Seed data has 5 queries
        assert result["total_tokens"] == 500  # 5 queries * 100 tokens each

    def test_delete_pipeline_results(self, service, mock_generate_func, cleanup_pipeline_results):
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
            assert len(results_before) == 5

        deleted_count = service.delete_pipeline_results(pipeline_id)
        assert deleted_count == 5

        # Verify results are deleted
        with service._create_uow() as uow:
            results_after = uow.executor_results.get_by_pipeline_id(pipeline_id)
            assert len(results_after) == 0

    def test_generation_result_without_token_usage(self, service, cleanup_pipeline_results):
        def generate_func_no_tokens(query: str, top_k: int) -> GenerationResult:
            return GenerationResult(text=f"Answer: {query}", token_usage=None)

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

        assert result["total_tokens"] is None

    def test_get_pipeline_config_not_found(self, service):
        config = service.get_pipeline_config(999999)
        assert config is None
