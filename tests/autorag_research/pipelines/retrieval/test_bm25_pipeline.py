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
    def pipeline(self, session_factory, cleanup_pipeline_results):
        pipeline = BM25RetrievalPipeline(
            session_factory=session_factory,
            name="test_bm25_pipeline",
            index_path=str(TEST_INDEX_PATH),
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)
        return pipeline

    def test_run(self, pipeline):
        result = pipeline.run(
            metric_id=SEED_METRIC_ID,
            top_k=3,
        )

        assert "pipeline_id" in result
        assert "metric_id" in result
        assert "total_queries" in result
        assert "total_results" in result
        assert result["pipeline_id"] == pipeline.pipeline_id
        assert result["metric_id"] == SEED_METRIC_ID
        assert result["total_queries"] == 5
        assert result["total_results"] == 15
