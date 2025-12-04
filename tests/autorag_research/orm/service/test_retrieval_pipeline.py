import pytest

from autorag_research.orm.service.retrieval_pipeline import RetrievalPipelineService

SEED_METRIC_ID = 1


class TestRetrievalPipelineService:
    @pytest.fixture
    def service(self, session_factory):
        return RetrievalPipelineService(session_factory)

    @pytest.fixture
    def mock_retrieval_func(self):
        def retrieval_func(queries: list[str], top_k: int) -> list[list[dict]]:
            results = []
            for _ in queries:
                results.append(
                    [
                        {"doc_id": 1, "score": 0.9},
                        {"doc_id": 2, "score": 0.8},
                    ][:top_k]
                )
            return results

        return retrieval_func

    def test_run(self, service, mock_retrieval_func):
        pipeline_id = service.create_pipeline(
            name="test_pipeline",
            config={"type": "test"},
        )

        result = service.run(
            retrieval_func=mock_retrieval_func,
            pipeline_id=pipeline_id,
            metric_id=SEED_METRIC_ID,
            top_k=2,
        )

        assert "pipeline_id" in result
        assert "metric_id" in result
        assert "total_queries" in result
        assert "total_results" in result
        assert result["pipeline_id"] == pipeline_id
        assert result["metric_id"] == SEED_METRIC_ID
        assert result["total_queries"] == 5
        assert result["total_results"] == 10

        service.delete_pipeline_results(pipeline_id)

    def test_delete_pipeline_results(self, service, mock_retrieval_func):
        pipeline_id = service.create_pipeline(
            name="test_delete_pipeline",
            config={"type": "test"},
        )

        service.run(
            retrieval_func=mock_retrieval_func,
            pipeline_id=pipeline_id,
            metric_id=SEED_METRIC_ID,
            top_k=2,
        )

        # Use UoW to verify results before deletion
        with service._create_uow() as uow:
            results_before = uow.chunk_results.get_by_pipeline(pipeline_id)
            assert len(results_before) > 0

        deleted_count = service.delete_pipeline_results(pipeline_id)

        assert deleted_count == 10

        # Use UoW to verify results after deletion
        with service._create_uow() as uow:
            results_after = uow.chunk_results.get_by_pipeline(pipeline_id)
            assert len(results_after) == 0
