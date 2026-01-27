"""Test cases for BM25Module.

Tests the VectorChord-BM25 based BM25 retrieval module.
These tests require the vchord_bm25 and pg_tokenizer extensions to be installed.
"""

import pytest
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.nodes.retrieval.bm25 import BM25Module
from autorag_research.orm.repository.chunk import ChunkRepository


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

        # Run search against seed data (chunks have contents like "Chunk 1-1", "Chunk 2-1", etc.)
        results = bm25_module.run(["Chunk"], top_k=3)

        assert len(results) == 1  # One query
        assert len(results[0]) <= 3  # Up to top_k results
        assert all("doc_id" in r and "score" in r and "content" in r for r in results[0])

        # Scores should be positive (negated BM25 scores)
        if results[0]:
            assert all(r["score"] > 0 for r in results[0])

    def test_run_multiple_queries(
        self,
        bm25_module: BM25Module,
        db_session: Session,
    ):
        """Test running with multiple queries using pre-seeded chunks."""
        # Populate BM25 vectors for existing seed data chunks
        ChunkRepository(db_session).batch_update_bm25_tokens(tokenizer="bert")

        # Run search with multiple queries
        results = bm25_module.run(["Chunk", "table"], top_k=2)

        assert len(results) == 2  # Two queries
        for query_results in results:
            assert len(query_results) <= 2
