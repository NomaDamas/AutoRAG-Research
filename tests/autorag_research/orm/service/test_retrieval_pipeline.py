import pytest

from autorag_research.orm.repository.query import QueryRepository
from autorag_research.orm.service.retrieval_pipeline import RetrievalPipelineService


class TestRetrievalPipelineService:
    @pytest.fixture
    def service(self, session_factory):
        return RetrievalPipelineService(session_factory)

    @pytest.fixture
    def mock_retrieval_func(self):
        def retrieval_func(query_ids: list[int | str], top_k: int) -> list[list[dict]]:
            """Mock retrieval function that returns 2 results per query ID."""
            results = []
            for _ in query_ids:
                results.append(
                    [
                        {"doc_id": 1, "score": 0.9},
                        {"doc_id": 2, "score": 0.8},
                    ][:top_k]
                )
            return results

        return retrieval_func

    def test_run_pipeline(self, service, mock_retrieval_func, session_factory):
        # Count actual queries in database
        with session_factory() as session:
            query_repo = QueryRepository(session)
            query_count = query_repo.count()

        pipeline_id = service.save_pipeline(
            name="test_pipeline",
            config={"type": "test"},
        )

        result = service.run_pipeline(
            retrieval_func=mock_retrieval_func,
            pipeline_id=pipeline_id,
            top_k=2,
        )

        assert "pipeline_id" in result
        assert "total_queries" in result
        assert "total_results" in result
        assert result["pipeline_id"] == pipeline_id
        assert result["total_queries"] == query_count
        assert result["total_results"] == query_count * 2

        service.delete_pipeline_results(pipeline_id)

    def test_delete_pipeline_results(self, service, mock_retrieval_func, session_factory):
        # Count actual queries in database
        with session_factory() as session:
            query_repo = QueryRepository(session)
            query_count = query_repo.count()

        pipeline_id = service.save_pipeline(
            name="test_delete_pipeline",
            config={"type": "test"},
        )

        service.run_pipeline(
            retrieval_func=mock_retrieval_func,
            pipeline_id=pipeline_id,
            top_k=2,
        )

        # Use UoW to verify results before deletion
        with service._create_uow() as uow:
            results_before = uow.chunk_results.get_by_pipeline(pipeline_id)
            assert len(results_before) > 0

        deleted_count = service.delete_pipeline_results(pipeline_id)

        expected_results = query_count * 2
        assert deleted_count == expected_results

        # Use UoW to verify results after deletion
        with service._create_uow() as uow:
            results_after = uow.chunk_results.get_by_pipeline(pipeline_id)
            assert len(results_after) == 0
