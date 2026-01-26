"""Test cases for BM25DBModule.

Tests the VectorChord-BM25 based BM25 retrieval module.
These tests require the vchord_bm25 and pg_tokenizer extensions to be installed.
"""

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.nodes.retrieval.bm25 import BM25Module
from autorag_research.orm.repository.chunk import ChunkRepository
from autorag_research.orm.schema import Chunk


def _check_bm25_extensions(session: Session) -> bool:
    """Check if VectorChord-BM25 extensions are available."""
    try:
        result = session.execute(text("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vchord_bm25')"))
        return bool(result.scalar())
    except Exception:
        return False


def _populate_bm25_tokens(session: Session, tokenizer: str = "bert") -> None:
    """Helper to populate BM25 vectors for test chunks."""
    repo = ChunkRepository(session)
    repo.batch_update_bm25_tokens(tokenizer=tokenizer)
    session.commit()


@pytest.fixture
def bm25_enabled(db_session: Session) -> bool:
    """Check if BM25 extensions are available."""
    return _check_bm25_extensions(db_session)


@pytest.fixture
def bm25_module(session_factory: sessionmaker[Session]) -> BM25Module:
    """Create a BM25DBModule instance for testing."""
    return BM25Module(
        session_factory=session_factory,
        tokenizer="bert",
        index_name="idx_chunk_bm25",
    )


class TestBM25DBModule:
    """Tests for BM25DBModule."""

    def test_module_initialization(self, session_factory: sessionmaker[Session]):
        """Test BM25DBModule initialization."""
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

    @pytest.mark.skipif(
        "not config.getoption('--run-bm25', default=False)",
        reason="BM25 tests require --run-bm25 flag and VectorChord-BM25 extension",
    )
    def test_run_single_query(
        self,
        bm25_module: BM25Module,
        db_session: Session,
        bm25_enabled: bool,
    ):
        """Test running with a single query."""
        if not bm25_enabled:
            pytest.skip("VectorChord-BM25 extension not available")

        # Create test chunks
        test_chunks = [
            Chunk(id=910001, contents="Introduction to machine learning algorithms", parent_caption=None),
            Chunk(id=910002, contents="Deep learning and neural network architectures", parent_caption=None),
            Chunk(id=910003, contents="Cooking delicious Italian pasta recipes", parent_caption=None),
        ]
        db_session.add_all(test_chunks)
        db_session.commit()

        try:
            # Populate BM25 vectors
            _populate_bm25_tokens(db_session, "bert")

            # Run search
            results = bm25_module.run(["machine learning"], top_k=3)

            assert len(results) == 1  # One query
            assert len(results[0]) <= 3  # Up to top_k results
            assert all("doc_id" in r and "score" in r and "content" in r for r in results[0])

            # Scores should be positive (negated BM25 scores)
            if results[0]:
                assert all(r["score"] > 0 for r in results[0])

        finally:
            # Cleanup
            for chunk in test_chunks:
                db_chunk = db_session.get(Chunk, chunk.id)
                if db_chunk:
                    db_session.delete(db_chunk)
            db_session.commit()

    @pytest.mark.skipif(
        "not config.getoption('--run-bm25', default=False)",
        reason="BM25 tests require --run-bm25 flag and VectorChord-BM25 extension",
    )
    def test_run_multiple_queries(
        self,
        bm25_module: BM25Module,
        db_session: Session,
        bm25_enabled: bool,
    ):
        """Test running with multiple queries."""
        if not bm25_enabled:
            pytest.skip("VectorChord-BM25 extension not available")

        # Create test chunks
        test_chunks = [
            Chunk(id=910011, contents="Python programming basics and syntax", parent_caption=None),
            Chunk(id=910012, contents="JavaScript web development tutorial", parent_caption=None),
            Chunk(id=910013, contents="SQL database queries and optimization", parent_caption=None),
        ]
        db_session.add_all(test_chunks)
        db_session.commit()

        try:
            # Populate BM25 vectors
            _populate_bm25_tokens(db_session, "bert")

            # Run search with multiple queries
            results = bm25_module.run(["python programming", "database"], top_k=2)

            assert len(results) == 2  # Two queries
            for query_results in results:
                assert len(query_results) <= 2

        finally:
            # Cleanup
            for chunk in test_chunks:
                db_chunk = db_session.get(Chunk, chunk.id)
                if db_chunk:
                    db_session.delete(db_chunk)
            db_session.commit()
