import pytest

from autorag_research.orm.repository.chunk_retrieved_result import ChunkRetrievedResultRepository
from autorag_research.orm.repository.metric import MetricRepository
from autorag_research.orm.repository.pipeline import PipelineRepository
from autorag_research.orm.schema import Pipeline
from autorag_research.orm.service.retrieval_pipeline import RetrievalPipelineService


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
                        {"doc_id": "1", "score": 0.9},
                        {"doc_id": "2", "score": 0.8},
                        {"doc_id": "3", "score": 0.7},
                    ][:top_k]
                )
            return results

        return retrieval_func

    @pytest.fixture
    def cleanup_pipeline_and_metric(self, session_factory):
        created_pipeline_ids = []
        created_metric_ids = []

        yield created_pipeline_ids, created_metric_ids

        session = session_factory()
        try:
            result_repo = ChunkRetrievedResultRepository(session)
            pipeline_repo = PipelineRepository(session)
            metric_repo = MetricRepository(session)

            for pipeline_id in created_pipeline_ids:
                result_repo.delete_by_pipeline(pipeline_id)
                pipeline = pipeline_repo.get_by_id(pipeline_id)
                if pipeline:
                    pipeline_repo.delete(pipeline)

            for metric_id in created_metric_ids:
                metric = metric_repo.get_by_id(metric_id)
                if metric:
                    metric_repo.delete(metric)

            session.commit()
        finally:
            session.close()

    def test_get_or_create_pipeline_creates_new(self, service, cleanup_pipeline_and_metric):
        created_pipeline_ids, _ = cleanup_pipeline_and_metric
        pipeline_name = "test_pipeline_new"
        pipeline_config = {"type": "test", "param": 1}

        pipeline_id = service._get_or_create_pipeline(name=pipeline_name, config=pipeline_config)
        created_pipeline_ids.append(pipeline_id)

        assert isinstance(pipeline_id, int)
        assert pipeline_id > 0

        with service._session_scope() as session:
            pipeline_repo = PipelineRepository(session)
            pipeline = pipeline_repo.get_by_name(pipeline_name)
            assert pipeline is not None
            assert pipeline.name == pipeline_name
            assert pipeline.config == pipeline_config

    def test_get_or_create_pipeline_returns_existing(self, service):
        pipeline_id = service._get_or_create_pipeline(name="baseline", config={"k": 5})

        assert pipeline_id == 1

    def test_get_or_create_metric_creates_new(self, service, cleanup_pipeline_and_metric):
        _, created_metric_ids = cleanup_pipeline_and_metric
        metric_name = "test_metric_new"
        metric_type = "retrieval"

        metric_id = service._get_or_create_metric(name=metric_name, metric_type=metric_type)
        created_metric_ids.append(metric_id)

        assert isinstance(metric_id, int)
        assert metric_id > 0

        with service._session_scope() as session:
            metric_repo = MetricRepository(session)
            metric = metric_repo.get_by_name_and_type(metric_name, metric_type)
            assert metric is not None
            assert metric.name == metric_name
            assert metric.type == metric_type

    def test_get_or_create_metric_returns_existing(self, service):
        metric_id = service._get_or_create_metric(name="retrieval@k", metric_type="retrieval")

        assert metric_id == 1

    def test_run_pipeline_with_mock_retrieval(self, service, mock_retrieval_func, cleanup_pipeline_and_metric):
        created_pipeline_ids, created_metric_ids = cleanup_pipeline_and_metric
        pipeline_name = "test_bm25_pipeline"
        pipeline_config = {"type": "bm25", "index": "test"}

        result = service.run(
            retrieval_func=mock_retrieval_func,
            pipeline_name=pipeline_name,
            pipeline_config=pipeline_config,
            metric_name="test_retrieval_metric",
            top_k=3,
            batch_size=10,
        )

        created_pipeline_ids.append(result["pipeline_id"])
        created_metric_ids.append(result["metric_id"])

        assert "pipeline_id" in result
        assert "metric_id" in result
        assert "total_queries" in result
        assert "total_results" in result
        assert result["total_queries"] == 5
        assert result["total_results"] == 15

    def test_run_pipeline_with_doc_id_mapping(self, service, cleanup_pipeline_and_metric):
        created_pipeline_ids, created_metric_ids = cleanup_pipeline_and_metric

        def retrieval_with_string_ids(queries: list[str], top_k: int) -> list[list[dict]]:
            results = []
            for _ in queries:
                results.append(
                    [
                        {"doc_id": "doc_a", "score": 0.9},
                        {"doc_id": "doc_b", "score": 0.8},
                    ][:top_k]
                )
            return results

        doc_id_mapping = {"doc_a": 1, "doc_b": 2}

        result = service.run(
            retrieval_func=retrieval_with_string_ids,
            pipeline_name="test_mapping_pipeline",
            pipeline_config={"type": "test"},
            metric_name="test_mapping_metric",
            top_k=2,
            doc_id_to_chunk_id=doc_id_mapping,
        )

        created_pipeline_ids.append(result["pipeline_id"])
        created_metric_ids.append(result["metric_id"])

        assert result["total_results"] == 10

    def test_run_pipeline_skips_unmapped_doc_ids(self, service, cleanup_pipeline_and_metric):
        created_pipeline_ids, created_metric_ids = cleanup_pipeline_and_metric

        def retrieval_with_unmapped_ids(queries: list[str], top_k: int) -> list[list[dict]]:
            results = []
            for _ in queries:
                results.append(
                    [
                        {"doc_id": "mapped_doc", "score": 0.9},
                        {"doc_id": "unmapped_doc", "score": 0.8},
                    ][:top_k]
                )
            return results

        doc_id_mapping = {"mapped_doc": 1}

        result = service.run(
            retrieval_func=retrieval_with_unmapped_ids,
            pipeline_name="test_skip_unmapped_pipeline",
            pipeline_config={"type": "test"},
            metric_name="test_skip_unmapped_metric",
            top_k=2,
            doc_id_to_chunk_id=doc_id_mapping,
        )

        created_pipeline_ids.append(result["pipeline_id"])
        created_metric_ids.append(result["metric_id"])

        assert result["total_results"] == 5

    def test_run_pipeline_skips_invalid_integer_doc_ids(self, service, cleanup_pipeline_and_metric):
        created_pipeline_ids, created_metric_ids = cleanup_pipeline_and_metric

        def retrieval_with_invalid_ids(queries: list[str], top_k: int) -> list[list[dict]]:
            results = []
            for _ in queries:
                results.append(
                    [
                        {"doc_id": "1", "score": 0.9},
                        {"doc_id": "not_an_int", "score": 0.8},
                    ][:top_k]
                )
            return results

        result = service.run(
            retrieval_func=retrieval_with_invalid_ids,
            pipeline_name="test_invalid_ids_pipeline",
            pipeline_config={"type": "test"},
            metric_name="test_invalid_ids_metric",
            top_k=2,
        )

        created_pipeline_ids.append(result["pipeline_id"])
        created_metric_ids.append(result["metric_id"])

        assert result["total_results"] == 5

    def test_delete_pipeline_results(self, service, mock_retrieval_func, cleanup_pipeline_and_metric):
        created_pipeline_ids, created_metric_ids = cleanup_pipeline_and_metric

        result = service.run(
            retrieval_func=mock_retrieval_func,
            pipeline_name="test_delete_pipeline",
            pipeline_config={"type": "test"},
            metric_name="test_delete_metric",
            top_k=2,
        )

        created_pipeline_ids.append(result["pipeline_id"])
        created_metric_ids.append(result["metric_id"])
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

    def test_run_pipeline_empty_results(self, service, cleanup_pipeline_and_metric):
        created_pipeline_ids, created_metric_ids = cleanup_pipeline_and_metric

        def empty_retrieval_func(queries: list[str], top_k: int) -> list[list[dict]]:
            return [[] for _ in queries]

        result = service.run(
            retrieval_func=empty_retrieval_func,
            pipeline_name="test_empty_pipeline",
            pipeline_config={"type": "test"},
            metric_name="test_empty_metric",
            top_k=10,
        )

        created_pipeline_ids.append(result["pipeline_id"])
        created_metric_ids.append(result["metric_id"])

        assert result["total_queries"] == 5
        assert result["total_results"] == 0

    def test_run_pipeline_batch_processing(self, service, cleanup_pipeline_and_metric):
        created_pipeline_ids, created_metric_ids = cleanup_pipeline_and_metric
        batch_counts = []

        def batch_tracking_retrieval(queries: list[str], top_k: int) -> list[list[dict]]:
            batch_counts.append(len(queries))
            return [[{"doc_id": "1", "score": 0.9}] for _ in queries]

        result = service.run(
            retrieval_func=batch_tracking_retrieval,
            pipeline_name="test_batch_pipeline",
            pipeline_config={"type": "test"},
            metric_name="test_batch_metric",
            top_k=1,
            batch_size=2,
        )

        created_pipeline_ids.append(result["pipeline_id"])
        created_metric_ids.append(result["metric_id"])

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
