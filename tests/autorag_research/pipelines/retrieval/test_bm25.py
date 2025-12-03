from pathlib import Path

import pytest

from autorag_research.orm.repository.chunk_retrieved_result import ChunkRetrievedResultRepository
from autorag_research.pipelines.retrieval.bm25 import BM25RetrievalPipeline

TEST_INDEX_PATH = Path(__file__).parent.parent.parent.parent / "resources" / "bm25_test_index"

SEED_PIPELINE_NAME = "baseline"
SEED_METRIC_NAME = "retrieval@k"


class TestBM25RetrievalPipeline:
    @pytest.fixture
    def cleanup_chunk_retrieved_results(self, session_factory):
        yield

        session = session_factory()
        try:
            result_repo = ChunkRetrievedResultRepository(session)
            result_repo.delete_by_pipeline(1)
            session.commit()
        finally:
            session.close()

    @pytest.fixture
    def pipeline_with_private_index(self, session_factory):
        return BM25RetrievalPipeline(
            session_factory=session_factory,
            index_path=str(TEST_INDEX_PATH),
        )

    def test_run_returns_correct_result_structure(self, pipeline_with_private_index, cleanup_chunk_retrieved_results):
        result = pipeline_with_private_index.run(
            pipeline_name=SEED_PIPELINE_NAME,
            metric_name=SEED_METRIC_NAME,
            top_k=3,
        )

        assert "pipeline_id" in result
        assert "metric_id" in result
        assert "total_queries" in result
        assert "total_results" in result
        assert result["pipeline_id"] == 1
        assert result["metric_id"] == 1
        assert result["total_queries"] == 5
        assert result["total_results"] == 15

    def test_run_with_batch_size(self, pipeline_with_private_index, cleanup_chunk_retrieved_results):
        result = pipeline_with_private_index.run(
            pipeline_name=SEED_PIPELINE_NAME,
            metric_name=SEED_METRIC_NAME,
            top_k=2,
            batch_size=2,
        )

        assert result["total_queries"] == 5
        assert result["total_results"] == 10

    def test_run_stores_retrieval_results_in_db(
        self, pipeline_with_private_index, session_factory, cleanup_chunk_retrieved_results
    ):
        result = pipeline_with_private_index.run(
            pipeline_name=SEED_PIPELINE_NAME,
            metric_name=SEED_METRIC_NAME,
            top_k=3,
        )

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

    def test_run_with_custom_bm25_parameters(self, session_factory, cleanup_chunk_retrieved_results):
        pipeline = BM25RetrievalPipeline(
            session_factory=session_factory,
            index_path=str(TEST_INDEX_PATH),
            k1=1.5,
            b=0.75,
        )

        result = pipeline.run(
            pipeline_name=SEED_PIPELINE_NAME,
            metric_name=SEED_METRIC_NAME,
            top_k=3,
        )

        assert result["pipeline_id"] == 1
        assert result["total_results"] == 15
