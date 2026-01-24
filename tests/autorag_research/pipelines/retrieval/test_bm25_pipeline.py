import pytest

from autorag_research.orm.repository.chunk_retrieved_result import ChunkRetrievedResultRepository
from autorag_research.pipelines.retrieval.bm25 import BM25RetrievalPipeline
from tests.autorag_research.pipelines.pipeline_test_utils import (
    PipelineTestConfig,
    PipelineTestVerifier,
)


class TestBM25RetrievalPipeline:
    @pytest.fixture
    def cleanup_pipeline_results(self, session_factory):
        """Cleanup fixture that deletes pipeline results after test."""
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
    def pipeline(self, session_factory, bm25_index_path, cleanup_pipeline_results):
        """Create a BM25RetrievalPipeline with real index from seed data."""
        pipeline = BM25RetrievalPipeline(
            session_factory=session_factory,
            name="test_bm25_pipeline",
            index_path=bm25_index_path,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)
        return pipeline

    def test_pipeline_creation(self, pipeline, bm25_index_path):
        """Test that pipeline is created correctly."""
        assert pipeline.pipeline_id > 0
        assert pipeline.index_path == bm25_index_path

    def test_pipeline_config(self, pipeline, bm25_index_path):
        """Test that pipeline config is correct."""
        config = pipeline._get_pipeline_config()
        assert config["type"] == "bm25"
        assert config["index_path"] == bm25_index_path
        assert config["k1"] == 0.9
        assert config["b"] == 0.4

    def test_retrieve_single_query(self, pipeline):
        """Test single query retrieval."""
        # Use a query that matches seed data chunks
        results = pipeline.retrieve("Chunk", top_k=3)

        assert isinstance(results, list)
        assert len(results) <= 3
        for result in results:
            assert "doc_id" in result
            assert "score" in result
            assert isinstance(result["doc_id"], int)

    def test_run(self, pipeline, session_factory):
        """Test running the full pipeline with verification."""
        # Seed data: 5 queries, 6 chunks
        # BM25 may not return results for all queries if they don't match
        result = pipeline.run(top_k=3)

        config = PipelineTestConfig(
            pipeline_type="retrieval",
            expected_total_queries=5,
            expected_min_results=0,  # BM25 may return 0 if no match
            check_persistence=True,
        )
        verifier = PipelineTestVerifier(result, pipeline.pipeline_id, session_factory, config)
        verifier.verify_all()

    def test_results_persisted_correctly(self, pipeline, session_factory):
        """Test that results are correctly persisted in database."""
        pipeline.run(top_k=3)

        session = session_factory()
        try:
            repo = ChunkRetrievedResultRepository(session)
            results = repo.get_by_pipeline(pipeline.pipeline_id)

            # Verify all results have valid chunk IDs (1-6 from seed data)
            valid_chunk_ids = {1, 2, 3, 4, 5, 6}
            for r in results:
                assert r.chunk_id in valid_chunk_ids
                assert r.rel_score >= 0  # BM25 scores are non-negative
        finally:
            session.close()
