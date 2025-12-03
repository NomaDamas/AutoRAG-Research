from pathlib import Path

import pytest

from autorag_research.orm.repository.chunk_retrieved_result import ChunkRetrievedResultRepository
from autorag_research.orm.repository.metric import MetricRepository
from autorag_research.orm.repository.pipeline import PipelineRepository
from autorag_research.pipelines.retrieval.bm25 import BM25RetrievalPipeline

TEST_INDEX_PATH = Path(__file__).parent.parent.parent.parent / "resources" / "bm25_test_index"


class TestBM25RetrievalPipeline:
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

    @pytest.fixture
    def pipeline_with_private_index(self, session_factory):
        return BM25RetrievalPipeline(
            session_factory=session_factory,
            index_path=str(TEST_INDEX_PATH),
        )

    @pytest.fixture
    def pipeline_with_prebuilt_index(self, session_factory):
        return BM25RetrievalPipeline(
            session_factory=session_factory,
            index_path="beir-v1.0.0-scifact.flat",
        )

    def test_init_default_parameters(self, session_factory):
        pipeline = BM25RetrievalPipeline(
            session_factory=session_factory,
            index_path=str(TEST_INDEX_PATH),
        )

        assert pipeline.index_path == str(TEST_INDEX_PATH)
        assert pipeline.k1 == 0.9
        assert pipeline.b == 0.4
        assert pipeline.language == "en"
        assert pipeline._service is not None

    def test_init_custom_parameters(self, session_factory):
        pipeline = BM25RetrievalPipeline(
            session_factory=session_factory,
            index_path=str(TEST_INDEX_PATH),
            k1=1.2,
            b=0.75,
            language="ko",
        )

        assert pipeline.index_path == str(TEST_INDEX_PATH)
        assert pipeline.k1 == 1.2
        assert pipeline.b == 0.75
        assert pipeline.language == "ko"

    def test_run_creates_pipeline_and_metric(self, pipeline_with_private_index, cleanup_pipeline_and_metric):
        created_pipeline_ids, created_metric_ids = cleanup_pipeline_and_metric

        result = pipeline_with_private_index.run(
            pipeline_name="test_bm25_pipeline_real",
            metric_name="test_bm25_metric_real",
            top_k=3,
        )

        created_pipeline_ids.append(result["pipeline_id"])
        created_metric_ids.append(result["metric_id"])

        assert "pipeline_id" in result
        assert "metric_id" in result
        assert "total_queries" in result
        assert "total_results" in result
        assert result["total_queries"] == 5
        assert result["total_results"] == 15

    def test_run_stores_correct_pipeline_config(self, session_factory, cleanup_pipeline_and_metric):
        created_pipeline_ids, created_metric_ids = cleanup_pipeline_and_metric

        pipeline = BM25RetrievalPipeline(
            session_factory=session_factory,
            index_path=str(TEST_INDEX_PATH),
            k1=0.9,
            b=0.4,
            language="en",
        )

        result = pipeline.run(
            pipeline_name="test_config_pipeline_real",
            metric_name="test_config_metric_real",
            top_k=10,
        )

        created_pipeline_ids.append(result["pipeline_id"])
        created_metric_ids.append(result["metric_id"])

        session = session_factory()
        try:
            pipeline_repo = PipelineRepository(session)
            stored_pipeline = pipeline_repo.get_by_id(result["pipeline_id"])

            assert stored_pipeline is not None
            assert stored_pipeline.config["type"] == "bm25"
            assert stored_pipeline.config["index_path"] == str(TEST_INDEX_PATH)
            assert stored_pipeline.config["top_k"] == 10
            assert stored_pipeline.config["k1"] == 0.9
            assert stored_pipeline.config["b"] == 0.4
            assert stored_pipeline.config["language"] == "en"
        finally:
            session.close()

    def test_run_with_doc_id_mapping(self, pipeline_with_private_index, cleanup_pipeline_and_metric):
        created_pipeline_ids, created_metric_ids = cleanup_pipeline_and_metric

        doc_id_mapping = {"0": 1, "1": 2, "2": 3}

        result = pipeline_with_private_index.run(
            pipeline_name="test_mapping_bm25_real",
            metric_name="test_mapping_bm25_metric_real",
            top_k=3,
            doc_id_to_chunk_id=doc_id_mapping,
        )

        created_pipeline_ids.append(result["pipeline_id"])
        created_metric_ids.append(result["metric_id"])

        assert result["total_queries"] == 5
        assert result["total_results"] > 0

    def test_run_with_batch_size(self, pipeline_with_private_index, cleanup_pipeline_and_metric):
        created_pipeline_ids, created_metric_ids = cleanup_pipeline_and_metric

        result = pipeline_with_private_index.run(
            pipeline_name="test_batch_bm25_real",
            metric_name="test_batch_bm25_metric_real",
            top_k=2,
            batch_size=2,
        )

        created_pipeline_ids.append(result["pipeline_id"])
        created_metric_ids.append(result["metric_id"])

        assert result["total_queries"] == 5
        assert result["total_results"] == 10

    def test_run_default_metric_name(self, pipeline_with_private_index, session_factory, cleanup_pipeline_and_metric):
        created_pipeline_ids, created_metric_ids = cleanup_pipeline_and_metric

        result = pipeline_with_private_index.run(
            pipeline_name="test_default_metric_bm25_real",
            top_k=3,
        )

        created_pipeline_ids.append(result["pipeline_id"])
        created_metric_ids.append(result["metric_id"])

        session = session_factory()
        try:
            metric_repo = MetricRepository(session)
            stored_metric = metric_repo.get_by_id(result["metric_id"])

            assert stored_metric is not None
            assert stored_metric.name == "bm25"
        finally:
            session.close()

    def test_run_reuses_existing_pipeline(
        self, pipeline_with_private_index, session_factory, cleanup_pipeline_and_metric
    ):
        created_pipeline_ids, created_metric_ids = cleanup_pipeline_and_metric

        result1 = pipeline_with_private_index.run(
            pipeline_name="test_reuse_pipeline_real",
            metric_name="test_reuse_metric_real",
            top_k=2,
        )

        created_pipeline_ids.append(result1["pipeline_id"])
        created_metric_ids.append(result1["metric_id"])

        session = session_factory()
        try:
            result_repo = ChunkRetrievedResultRepository(session)
            result_repo.delete_by_pipeline(result1["pipeline_id"])
            session.commit()
        finally:
            session.close()

        result2 = pipeline_with_private_index.run(
            pipeline_name="test_reuse_pipeline_real",
            metric_name="test_reuse_metric_real",
            top_k=2,
        )

        assert result1["pipeline_id"] == result2["pipeline_id"]
        assert result1["metric_id"] == result2["metric_id"]

    def test_run_stores_retrieval_results_in_db(
        self, pipeline_with_private_index, session_factory, cleanup_pipeline_and_metric
    ):
        created_pipeline_ids, created_metric_ids = cleanup_pipeline_and_metric

        result = pipeline_with_private_index.run(
            pipeline_name="test_results_stored_real",
            metric_name="test_results_metric_real",
            top_k=3,
        )

        created_pipeline_ids.append(result["pipeline_id"])
        created_metric_ids.append(result["metric_id"])

        session = session_factory()
        try:
            result_repo = ChunkRetrievedResultRepository(session)
            stored_results = result_repo.get_by_pipeline(result["pipeline_id"])

            assert len(stored_results) == result["total_results"]
            assert all(r.pipeline_id == result["pipeline_id"] for r in stored_results)
            assert all(r.metric_id == result["metric_id"] for r in stored_results)
            assert all(r.rel_score is not None for r in stored_results)
        finally:
            session.close()

    def test_run_with_custom_bm25_parameters(self, session_factory, cleanup_pipeline_and_metric):
        created_pipeline_ids, created_metric_ids = cleanup_pipeline_and_metric

        pipeline = BM25RetrievalPipeline(
            session_factory=session_factory,
            index_path=str(TEST_INDEX_PATH),
            k1=1.5,
            b=0.75,
        )

        result = pipeline.run(
            pipeline_name="test_custom_bm25_params_real",
            metric_name="test_custom_params_metric_real",
            top_k=3,
        )

        created_pipeline_ids.append(result["pipeline_id"])
        created_metric_ids.append(result["metric_id"])

        session = session_factory()
        try:
            pipeline_repo = PipelineRepository(session)
            stored_pipeline = pipeline_repo.get_by_id(result["pipeline_id"])

            assert stored_pipeline.config["k1"] == 1.5
            assert stored_pipeline.config["b"] == 0.75
        finally:
            session.close()
