"""Tests for BM25 retrieval module."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock pyserini module before importing BM25Module
# ruff: noqa: E402
mock_lucene_searcher = MagicMock()
sys.modules["pyserini"] = MagicMock()
sys.modules["pyserini.search"] = MagicMock()
sys.modules["pyserini.search.lucene"] = MagicMock()
sys.modules["pyserini.search.lucene"].LuceneSearcher = mock_lucene_searcher

from autorag_research.nodes.retrieval.bm25 import BM25Module


class MockHit:
    """Mock class for Pyserini search hits."""

    def __init__(self, docid: str, score: float, raw: str | None = None, contents: str | None = None):
        self.docid = docid
        self.score = score
        if raw is not None:
            self.raw = raw
        if contents is not None:
            self.contents = contents


@pytest.fixture
def mock_searcher():
    """Create a mock LuceneSearcher."""
    searcher = MagicMock()
    searcher.search.return_value = [
        MockHit("doc1", 2.5, raw="Document 1 content"),
        MockHit("doc2", 2.0, raw="Document 2 content"),
        MockHit("doc3", 1.5, raw="Document 3 content"),
    ]
    return searcher


class TestBM25ModuleInit:
    """Tests for BM25Module initialization."""

    def test_init_requires_index_name_or_path(self):
        """Should raise ValueError when neither index_name nor index_path is provided."""
        with pytest.raises(ValueError, match="Either index_name or index_path must be provided"):
            BM25Module()

    @patch("autorag_research.nodes.retrieval.bm25.LuceneSearcher")
    def test_init_with_index_path(self, mock_lucene_class):
        """Should initialize with custom index path."""
        mock_searcher = MagicMock()
        mock_lucene_class.return_value = mock_searcher

        module = BM25Module(index_path="/path/to/index")

        mock_lucene_class.assert_called_once_with("/path/to/index")
        mock_searcher.set_language.assert_called_once_with("en")
        mock_searcher.set_bm25.assert_called_once_with(k1=0.9, b=0.4)
        assert module.index_name is None
        assert module.index_path is not None

    @patch("autorag_research.nodes.retrieval.bm25.LuceneSearcher")
    def test_init_with_prebuilt_index(self, mock_lucene_class):
        """Should initialize with pre-built index."""
        mock_searcher = MagicMock()
        mock_lucene_class.from_prebuilt_index.return_value = mock_searcher

        module = BM25Module(index_name="msmarco-passage")

        mock_lucene_class.from_prebuilt_index.assert_called_once_with("msmarco-passage")
        mock_searcher.set_language.assert_called_once_with("en")
        mock_searcher.set_bm25.assert_called_once_with(k1=0.9, b=0.4)
        assert module.index_name == "msmarco-passage"
        assert module.index_path is None

    @patch("autorag_research.nodes.retrieval.bm25.LuceneSearcher")
    def test_init_with_custom_bm25_params(self, mock_lucene_class):
        """Should set custom BM25 parameters."""
        mock_searcher = MagicMock()
        mock_lucene_class.return_value = mock_searcher

        module = BM25Module(index_path="/path/to/index", k1=1.2, b=0.75)

        mock_searcher.set_bm25.assert_called_once_with(k1=1.2, b=0.75)
        assert module.k1 == 1.2
        assert module.b == 0.75

    @patch("autorag_research.nodes.retrieval.bm25.LuceneSearcher")
    def test_init_with_custom_language(self, mock_lucene_class):
        """Should set custom language."""
        mock_searcher = MagicMock()
        mock_lucene_class.return_value = mock_searcher

        BM25Module(index_path="/path/to/index", language="ko")

        mock_searcher.set_language.assert_called_once_with("ko")

    @patch("autorag_research.nodes.retrieval.bm25.LuceneSearcher")
    def test_init_prefers_index_name_over_path(self, mock_lucene_class):
        """When both index_name and index_path are provided, index_name takes precedence."""
        mock_searcher = MagicMock()
        mock_lucene_class.from_prebuilt_index.return_value = mock_searcher

        module = BM25Module(index_name="msmarco-passage", index_path="/path/to/index")

        mock_lucene_class.from_prebuilt_index.assert_called_once_with("msmarco-passage")
        mock_lucene_class.assert_not_called()
        assert module.index_name == "msmarco-passage"
        assert module.index_path is None


class TestBM25ModuleRun:
    """Tests for BM25Module run method."""

    @patch("autorag_research.nodes.retrieval.bm25.LuceneSearcher")
    def test_run_single_query(self, mock_lucene_class, mock_searcher):
        """Should return results for a single query."""
        mock_lucene_class.return_value = mock_searcher

        module = BM25Module(index_path="/path/to/index")
        results = module.run(["test query"], top_k=3)

        assert len(results) == 1
        assert len(results[0]) == 3
        assert results[0][0]["doc_id"] == "doc1"
        assert results[0][0]["score"] == 2.5
        assert results[0][0]["content"] == "Document 1 content"

    @patch("autorag_research.nodes.retrieval.bm25.LuceneSearcher")
    def test_run_multiple_queries(self, mock_lucene_class, mock_searcher):
        """Should return results for multiple queries."""
        mock_lucene_class.return_value = mock_searcher

        module = BM25Module(index_path="/path/to/index")
        results = module.run(["query 1", "query 2", "query 3"], top_k=3)

        assert len(results) == 3
        assert mock_searcher.search.call_count == 3

    @patch("autorag_research.nodes.retrieval.bm25.LuceneSearcher")
    def test_run_with_custom_top_k(self, mock_lucene_class, mock_searcher):
        """Should pass correct top_k to searcher."""
        mock_lucene_class.return_value = mock_searcher

        module = BM25Module(index_path="/path/to/index")
        module.run(["test query"], top_k=5)

        mock_searcher.search.assert_called_with("test query", k=5)

    @patch("autorag_research.nodes.retrieval.bm25.LuceneSearcher")
    def test_run_empty_query_list(self, mock_lucene_class, mock_searcher):
        """Should return empty list for empty query list."""
        mock_lucene_class.return_value = mock_searcher

        module = BM25Module(index_path="/path/to/index")
        results = module.run([], top_k=10)

        assert results == []
        mock_searcher.search.assert_not_called()

    @patch("autorag_research.nodes.retrieval.bm25.LuceneSearcher")
    def test_run_uses_contents_when_raw_not_available(self, mock_lucene_class):
        """Should use contents attribute when raw is not available."""
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = [
            MockHit("doc1", 2.5, contents="Document content via contents"),
        ]
        mock_lucene_class.return_value = mock_searcher

        module = BM25Module(index_path="/path/to/index")
        results = module.run(["test query"], top_k=1)

        assert results[0][0]["content"] == "Document content via contents"

    @patch("autorag_research.nodes.retrieval.bm25.LuceneSearcher")
    def test_run_empty_content_when_neither_raw_nor_contents(self, mock_lucene_class):
        """Should return empty content when neither raw nor contents is available."""
        mock_hit = MagicMock()
        mock_hit.docid = "doc1"
        mock_hit.score = 2.5
        del mock_hit.raw
        del mock_hit.contents

        mock_searcher = MagicMock()
        mock_searcher.search.return_value = [mock_hit]
        mock_lucene_class.return_value = mock_searcher

        module = BM25Module(index_path="/path/to/index")
        results = module.run(["test query"], top_k=1)

        assert results[0][0]["content"] == ""

    @patch("autorag_research.nodes.retrieval.bm25.LuceneSearcher")
    def test_run_no_results(self, mock_lucene_class):
        """Should handle queries with no results."""
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = []
        mock_lucene_class.return_value = mock_searcher

        module = BM25Module(index_path="/path/to/index")
        results = module.run(["query with no results"], top_k=10)

        assert len(results) == 1
        assert results[0] == []

    @patch("autorag_research.nodes.retrieval.bm25.LuceneSearcher")
    def test_run_default_top_k(self, mock_lucene_class, mock_searcher):
        """Should use default top_k of 10."""
        mock_lucene_class.return_value = mock_searcher

        module = BM25Module(index_path="/path/to/index")
        module.run(["test query"])

        mock_searcher.search.assert_called_with("test query", k=10)

    @patch("autorag_research.nodes.retrieval.bm25.LuceneSearcher")
    def test_run_result_structure(self, mock_lucene_class, mock_searcher):
        """Should return results with correct structure."""
        mock_lucene_class.return_value = mock_searcher

        module = BM25Module(index_path="/path/to/index")
        results = module.run(["test query"], top_k=3)

        for result in results[0]:
            assert "doc_id" in result
            assert "score" in result
            assert "content" in result
            assert isinstance(result["doc_id"], str)
            assert isinstance(result["score"], float)
            assert isinstance(result["content"], str)
