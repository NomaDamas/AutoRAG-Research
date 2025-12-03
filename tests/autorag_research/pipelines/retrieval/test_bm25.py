from pathlib import Path

import pytest

from autorag_research.orm.repository.chunk_retrieved_result import ChunkRetrievedResultRepository
from autorag_research.pipelines.retrieval.bm25 import BM25RetrievalPipeline

TEST_INDEX_PATH = Path(__file__).parent.parent.parent.parent / "resources" / "bm25_test_index"

SEED_METRIC_ID = 1


class TestBM25RetrievalPipeline:
    @pytest.fixture
    def cleanup_pipeline_results(self, session_factory):
        created_pipeline_ids = []

        yield created_pipeline_ids

        session = session_factory()
        try:
            result_repo = ChunkRetrievedResultRepository(session)
            for pipeline_id in created_pipeline_ids:
                result_repo.delete_by_pipeline(pipeline_id)
            session.commit()
        finally:
            session.close()

    @pytest.fixture
    def pipeline_with_private_index(self, session_factory):
        return BM25RetrievalPipeline(
            session_factory=session_factory,
            index_path=str(TEST_INDEX_PATH),
        )

    def test_run_returns_correct_result_structure(self, pipeline_with_private_index, cleanup_pipeline_results):
        result = pipeline_with_private_index.run(
            metric_id=SEED_METRIC_ID,
            top_k=3,
        )

        cleanup_pipeline_results.append(result["pipeline_id"])

        assert "pipeline_id" in result
        assert "metric_id" in result
        assert "total_queries" in result
        assert "total_results" in result
        assert result["metric_id"] == SEED_METRIC_ID
        assert result["total_queries"] == 5
        assert result["total_results"] == 15

    def test_run_with_batch_size(self, pipeline_with_private_index, cleanup_pipeline_results):
        result = pipeline_with_private_index.run(
            metric_id=SEED_METRIC_ID,
            top_k=2,
            batch_size=2,
        )

        cleanup_pipeline_results.append(result["pipeline_id"])

        assert result["total_queries"] == 5
        assert result["total_results"] == 10

    def test_run_stores_retrieval_results_in_db(
        self, pipeline_with_private_index, session_factory, cleanup_pipeline_results
    ):
        result = pipeline_with_private_index.run(
            metric_id=SEED_METRIC_ID,
            top_k=3,
        )

        cleanup_pipeline_results.append(result["pipeline_id"])

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

    def test_run_with_custom_bm25_parameters(self, session_factory, cleanup_pipeline_results):
        pipeline = BM25RetrievalPipeline(
            session_factory=session_factory,
            index_path=str(TEST_INDEX_PATH),
            k1=1.5,
            b=0.75,
        )

        result = pipeline.run(
            metric_id=SEED_METRIC_ID,
            top_k=3,
        )

        cleanup_pipeline_results.append(result["pipeline_id"])

        assert result["total_results"] == 15
