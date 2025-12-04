from unittest.mock import MagicMock, patch

import pytest

from autorag_research.orm.repository.chunk_retrieved_result import ChunkRetrievedResultRepository
from autorag_research.pipelines.retrieval.bm25 import BM25RetrievalPipeline


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
    def mock_bm25_module(self):
        def mock_run(queries: list[str], top_k: int) -> list[list[dict]]:
            results = []
            for _ in queries:
                results.append([{"doc_id": i + 1, "score": 0.9 - i * 0.1} for i in range(top_k)])
            return results

        mock = MagicMock()
        mock.run = mock_run
        return mock

    @pytest.fixture
    def pipeline(self, session_factory, cleanup_pipeline_results):
        pipeline = BM25RetrievalPipeline(
            session_factory=session_factory,
            name="test_bm25_pipeline",
            index_path="/fake/index/path",
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)
        return pipeline

    def test_run(self, pipeline, mock_bm25_module):
        with patch(
            "autorag_research.pipelines.retrieval.bm25.BM25Module",
            return_value=mock_bm25_module,
        ):
            result = pipeline.run(top_k=3)

        assert "pipeline_id" in result
        assert "total_queries" in result
        assert "total_results" in result
        assert result["pipeline_id"] == pipeline.pipeline_id
        assert result["total_queries"] == 5
        assert result["total_results"] == 15
