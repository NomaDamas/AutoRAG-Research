import pytest

from autorag_research.orm.repository.chunk_retrieved_result import ChunkRetrievedResultRepository
from autorag_research.orm.repository.pipeline import PipelineRepository
from autorag_research.orm.schema import Pipeline
from autorag_research.orm.service.retrieval_pipeline import RetrievalPipelineService

SEED_PIPELINE_NAME = "baseline"
SEED_PIPELINE_ID = 1
SEED_METRIC_NAME = "retrieval@k"
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
    def cleanup_chunk_retrieved_results(self, session_factory):
        yield

        session = session_factory()
        try:
            result_repo = ChunkRetrievedResultRepository(session)
            result_repo.delete_by_pipeline(SEED_PIPELINE_ID)
            session.commit()
        finally:
            session.close()

    def test_get_or_create_pipeline_returns_existing(self, service):
        pipeline_id = service._get_or_create_pipeline(name=SEED_PIPELINE_NAME, config={"k": 5})

        assert pipeline_id == SEED_PIPELINE_ID

    def test_get_or_create_metric_returns_existing(self, service):
        metric_id = service._get_or_create_metric(name=SEED_METRIC_NAME, metric_type="retrieval")

        assert metric_id == SEED_METRIC_ID

    def test_run_pipeline_with_mock_retrieval(self, service, mock_retrieval_func, cleanup_chunk_retrieved_results):
        result = service.run(
            retrieval_func=mock_retrieval_func,
            pipeline_name=SEED_PIPELINE_NAME,
            pipeline_config={"type": "bm25", "index": "test"},
            metric_name=SEED_METRIC_NAME,
            top_k=3,
            batch_size=10,
        )

        assert "pipeline_id" in result
        assert "metric_id" in result
        assert "total_queries" in result
        assert "total_results" in result
        assert result["pipeline_id"] == SEED_PIPELINE_ID
        assert result["metric_id"] == SEED_METRIC_ID
        assert result["total_queries"] == 5
        assert result["total_results"] == 15

    def test_delete_pipeline_results(self, service, mock_retrieval_func, cleanup_chunk_retrieved_results):
        result = service.run(
            retrieval_func=mock_retrieval_func,
            pipeline_name=SEED_PIPELINE_NAME,
            pipeline_config={"type": "test"},
            metric_name=SEED_METRIC_NAME,
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

    def test_run_pipeline_empty_results(self, service, cleanup_chunk_retrieved_results):
        def empty_retrieval_func(queries: list[str], top_k: int) -> list[list[dict]]:
            return [[] for _ in queries]

        result = service.run(
            retrieval_func=empty_retrieval_func,
            pipeline_name=SEED_PIPELINE_NAME,
            pipeline_config={"type": "test"},
            metric_name=SEED_METRIC_NAME,
            top_k=10,
        )

        assert result["total_queries"] == 5
        assert result["total_results"] == 0

    def test_run_pipeline_batch_processing(self, service, cleanup_chunk_retrieved_results):
        batch_counts = []

        def batch_tracking_retrieval(queries: list[str], top_k: int) -> list[list[dict]]:
            batch_counts.append(len(queries))
            return [[{"doc_id": 1, "score": 0.9}] for _ in queries]

        result = service.run(
            retrieval_func=batch_tracking_retrieval,
            pipeline_name=SEED_PIPELINE_NAME,
            pipeline_config={"type": "test"},
            metric_name=SEED_METRIC_NAME,
            top_k=1,
            batch_size=2,
        )

        assert result["total_queries"] == 5
        assert sum(batch_counts) == 5
        assert all(count <= 2 for count in batch_counts)

    def test_session_scope_rollback_on_error(self, service):
        with pytest.raises(ValueError), service._session_scope() as session:
            pipeline_repo = PipelineRepository(session)
            pipeline_repo.add(Pipeline(name="rollback_test", config={}))
            raise ValueError("Test error")  # noqa: TRY003

        with service._session_scope() as session:
            pipeline_repo = PipelineRepository(session)
            pipeline = pipeline_repo.get_by_name("rollback_test")
            assert pipeline is None
