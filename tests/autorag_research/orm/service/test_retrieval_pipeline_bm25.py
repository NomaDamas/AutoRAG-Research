"""Test cases for RetrievalPipelineService.bm25_search method.

Tests the BM25 search functionality moved from BM25Module to the service layer.
These tests require the vchord_bm25 and pg_tokenizer extensions to be installed.
"""

import pytest
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.orm.repository.chunk import ChunkRepository
from autorag_research.orm.schema import Query
from autorag_research.orm.service.retrieval_pipeline import RetrievalPipelineService


@pytest.fixture
def retrieval_service(session_factory: sessionmaker[Session]) -> RetrievalPipelineService:
    """Create a RetrievalPipelineService instance for testing."""
    return RetrievalPipelineService(session_factory=session_factory)


class TestBM25Search:
    """Tests for RetrievalPipelineService.bm25_search()."""

    def test_bm25_search_empty_queries(self, retrieval_service: RetrievalPipelineService):
        """Test bm25_search with empty query list."""
        results = retrieval_service.bm25_search([], top_k=5)
        assert results == []

    def test_bm25_search_single_query(
        self,
        retrieval_service: RetrievalPipelineService,
        db_session: Session,
    ):
        """Test bm25_search with a single query using pre-seeded chunks."""
        # Populate BM25 vectors for existing seed data chunks
        ChunkRepository(db_session).batch_update_bm25_tokens(tokenizer="bert")

        # Create a test query
        test_query = Query(contents="Chunk")
        db_session.add(test_query)
        db_session.commit()

        try:
            # Run search against seed data (chunks have contents like "Chunk 1-1", "Chunk 2-1", etc.)
            results = retrieval_service.bm25_search([test_query.id], top_k=3)

            assert len(results) == 1  # One query
            assert len(results[0]) <= 3  # Up to top_k results
            assert all("doc_id" in r and "score" in r and "content" in r for r in results[0])

            # Scores should be positive (negated BM25 scores)
            if results[0]:
                assert all(r["score"] > 0 for r in results[0])
        finally:
            # Cleanup
            db_session.delete(test_query)
            db_session.commit()

    def test_bm25_search_multiple_queries(
        self,
        retrieval_service: RetrievalPipelineService,
        db_session: Session,
    ):
        """Test bm25_search with multiple queries using pre-seeded chunks."""
        # Populate BM25 vectors for existing seed data chunks
        ChunkRepository(db_session).batch_update_bm25_tokens(tokenizer="bert")

        # Create test queries
        test_queries = [
            Query(contents="Chunk"),
            Query(contents="table"),
        ]
        db_session.add_all(test_queries)
        db_session.commit()

        try:
            # Run search with multiple query IDs
            query_ids = [q.id for q in test_queries]
            results = retrieval_service.bm25_search(query_ids, top_k=2)

            assert len(results) == 2  # Two queries
            for query_results in results:
                assert len(query_results) <= 2
        finally:
            # Cleanup
            for q in test_queries:
                db_session.delete(q)
            db_session.commit()

    def test_bm25_search_query_not_found(self, retrieval_service: RetrievalPipelineService):
        """Test that ValueError is raised when query ID is not found."""
        with pytest.raises(ValueError, match="Query 999999 not found"):
            retrieval_service.bm25_search([999999], top_k=3)

    def test_bm25_search_custom_tokenizer(
        self,
        retrieval_service: RetrievalPipelineService,
        db_session: Session,
    ):
        """Test bm25_search with custom tokenizer parameter."""
        # Populate BM25 vectors
        ChunkRepository(db_session).batch_update_bm25_tokens(tokenizer="bert")

        # Create a test query
        test_query = Query(contents="Chunk")
        db_session.add(test_query)
        db_session.commit()

        try:
            # Run search with explicit tokenizer
            results = retrieval_service.bm25_search(
                [test_query.id],
                top_k=3,
                tokenizer="bert",
                index_name="idx_chunk_bm25",
            )

            assert len(results) == 1
            assert all("doc_id" in r and "score" in r and "content" in r for r in results[0])
        finally:
            db_session.delete(test_query)
            db_session.commit()
