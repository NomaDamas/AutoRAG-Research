import pytest

from autorag_research.orm.repository.chunk_retrieved_result import ChunkRetrievedResultRepository
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
                        {"doc_id": 3, "score": 0.7},
                    ][:top_k]
                )
            return results

        return retrieval_func

    @pytest.fixture
    def cleanup_pipeline_results(self, service):
        created_pipeline_ids = []

        yield created_pipeline_ids

        for pipeline_id in created_pipeline_ids:
            service.delete_pipeline_results(pipeline_id)

    def test_run_pipeline_creates_new_pipeline(self, service, mock_retrieval_func, cleanup_pipeline_results):
        result = service.run(
            retrieval_func=mock_retrieval_func,
            pipeline_config={"type": "bm25", "index": "test"},
            metric_id=SEED_METRIC_ID,
            top_k=3,
            batch_size=10,
        )

        cleanup_pipeline_results.append(result["pipeline_id"])

        assert "pipeline_id" in result
        assert "metric_id" in result
        assert "total_queries" in result
        assert "total_results" in result
        assert result["metric_id"] == SEED_METRIC_ID
        assert result["total_queries"] == 5
        assert result["total_results"] == 15

    def test_run_pipeline_creates_different_pipeline_each_time(
        self, service, mock_retrieval_func, cleanup_pipeline_results
    ):
        result1 = service.run(
            retrieval_func=mock_retrieval_func,
            pipeline_config={"type": "bm25"},
            metric_id=SEED_METRIC_ID,
            top_k=2,
        )
        cleanup_pipeline_results.append(result1["pipeline_id"])

        result2 = service.run(
            retrieval_func=mock_retrieval_func,
            pipeline_config={"type": "bm25"},
            metric_id=SEED_METRIC_ID,
            top_k=2,
        )
        cleanup_pipeline_results.append(result2["pipeline_id"])

        assert result1["pipeline_id"] != result2["pipeline_id"]

    def test_delete_pipeline_results(self, service, mock_retrieval_func, cleanup_pipeline_results):
        result = service.run(
            retrieval_func=mock_retrieval_func,
            pipeline_config={"type": "test"},
            metric_id=SEED_METRIC_ID,
            top_k=2,
        )

        pipeline_id = result["pipeline_id"]

        with service._session_scope() as session:
            result_repo = ChunkRetrievedResultRepository(session)
            results_before = result_repo.get_by_pipeline(pipeline_id)
            assert len(results_before) > 0

        deleted_count = service.delete_pipeline_results(pipeline_id)

        assert deleted_count == 10

        with service._session_scope() as session:
            result_repo = ChunkRetrievedResultRepository(session)
            results_after = result_repo.get_by_pipeline(pipeline_id)
            assert len(results_after) == 0

    def test_run_pipeline_empty_results(self, service, cleanup_pipeline_results):
        def empty_retrieval_func(queries: list[str], top_k: int) -> list[list[dict]]:
            return [[] for _ in queries]

        result = service.run(
            retrieval_func=empty_retrieval_func,
            pipeline_config={"type": "test"},
            metric_id=SEED_METRIC_ID,
            top_k=10,
        )

        cleanup_pipeline_results.append(result["pipeline_id"])

        assert result["total_queries"] == 5
        assert result["total_results"] == 0

    def test_run_pipeline_batch_processing(self, service, cleanup_pipeline_results):
        batch_counts = []

        def batch_tracking_retrieval(queries: list[str], top_k: int) -> list[list[dict]]:
            batch_counts.append(len(queries))
            return [[{"doc_id": 1, "score": 0.9}] for _ in queries]

        result = service.run(
            retrieval_func=batch_tracking_retrieval,
            pipeline_config={"type": "test"},
            metric_id=SEED_METRIC_ID,
            top_k=1,
            batch_size=2,
        )

        cleanup_pipeline_results.append(result["pipeline_id"])

        assert result["total_queries"] == 5
        assert sum(batch_counts) == 5
        assert all(count <= 2 for count in batch_counts)
