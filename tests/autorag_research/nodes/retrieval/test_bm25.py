"""Test cases for BM25Module.

Tests the VectorChord-BM25 based BM25 retrieval module.
These tests require the vchord_bm25 and pg_tokenizer extensions to be installed.
"""

import pytest
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.nodes.retrieval.bm25 import BM25Module
from autorag_research.orm.repository.chunk import ChunkRepository
from autorag_research.orm.schema import Query


@pytest.fixture
def bm25_module(session_factory: sessionmaker[Session]) -> BM25Module:
    """Create a BM25Module instance for testing."""
    return BM25Module(
        session_factory=session_factory,
        tokenizer="bert",
        index_name="idx_chunk_bm25",
    )


class TestBM25Module:
    """Tests for BM25Module."""

    def test_module_initialization(self, session_factory: sessionmaker[Session]):
        """Test BM25Module initialization."""
        module = BM25Module(
            session_factory=session_factory,
            tokenizer="bert",
            index_name="idx_chunk_bm25",
        )

        assert module.tokenizer == "bert"
        assert module.index_name == "idx_chunk_bm25"
        assert module.session_factory is not None

    def test_run_empty_queries(self, bm25_module: BM25Module):
        """Test running with empty query list."""
        results = bm25_module.run([], top_k=5)
        assert results == []

    def test_run_single_query(
        self,
        bm25_module: BM25Module,
        db_session: Session,
    ):
        """Test running with a single query using pre-seeded chunks."""
        # Populate BM25 vectors for existing seed data chunks
        ChunkRepository(db_session).batch_update_bm25_tokens(tokenizer="bert")

        # Create a test query
        test_query = Query(contents="Chunk")
        db_session.add(test_query)
        db_session.commit()

        try:
            # Run search against seed data (chunks have contents like "Chunk 1-1", "Chunk 2-1", etc.)
            results = bm25_module.run([test_query.id], top_k=3)

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

    def test_run_multiple_queries(
        self,
        bm25_module: BM25Module,
        db_session: Session,
    ):
        """Test running with multiple queries using pre-seeded chunks."""
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
            results = bm25_module.run(query_ids, top_k=2)

            assert len(results) == 2  # Two queries
            for query_results in results:
                assert len(query_results) <= 2
        finally:
            # Cleanup
            for q in test_queries:
                db_session.delete(q)
            db_session.commit()


class TestBM25ModuleSearch:
    """Tests for BM25Module.search() method - direct text input."""

    def test_search_empty_texts(self, bm25_module: BM25Module):
        """Test search with empty text list."""
        results = bm25_module.search([], top_k=5)
        assert results == []

    def test_search_single_text(
        self,
        bm25_module: BM25Module,
        db_session: Session,
    ):
        """Test search with a single text query."""
        # Populate BM25 vectors for existing seed data chunks
        ChunkRepository(db_session).batch_update_bm25_tokens(tokenizer="bert")

        # Run search directly with text (bypassing Query table)
        results = bm25_module.search(["Chunk"], top_k=3)

        assert len(results) == 1  # One query
        assert len(results[0]) <= 3  # Up to top_k results
        assert all("doc_id" in r and "score" in r and "content" in r for r in results[0])

        # Scores should be positive
        if results[0]:
            assert all(r["score"] > 0 for r in results[0])

    def test_search_multiple_texts(
        self,
        bm25_module: BM25Module,
        db_session: Session,
    ):
        """Test search with multiple text queries."""
        # Populate BM25 vectors for existing seed data chunks
        ChunkRepository(db_session).batch_update_bm25_tokens(tokenizer="bert")

        # Run search with multiple texts
        results = bm25_module.search(["Chunk", "table"], top_k=2)

        assert len(results) == 2  # Two queries
        for query_results in results:
            assert len(query_results) <= 2

    def test_search_vs_run_equivalence(
        self,
        bm25_module: BM25Module,
        db_session: Session,
    ):
        """Test that search() and run() produce equivalent results for same text."""
        # Populate BM25 vectors for existing seed data chunks
        ChunkRepository(db_session).batch_update_bm25_tokens(tokenizer="bert")

        # Create a test query with specific text
        test_query = Query(contents="Chunk")
        db_session.add(test_query)
        db_session.commit()

        try:
            # Get results via run() (using query ID)
            run_results = bm25_module.run([test_query.id], top_k=3)

            # Get results via search() (using direct text)
            search_results = bm25_module.search(["Chunk"], top_k=3)

            # Results should be equivalent
            assert len(run_results) == len(search_results)
            assert len(run_results[0]) == len(search_results[0])

            # Same doc_ids and scores
            for run_r, search_r in zip(run_results[0], search_results[0], strict=True):
                assert run_r["doc_id"] == search_r["doc_id"]
                assert run_r["score"] == pytest.approx(search_r["score"])
        finally:
            db_session.delete(test_query)
            db_session.commit()
