"""Test cases for CohereReranker.

Tests the Cohere Rerank API based reranker module.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autorag_research.exceptions import CohereAPIKeyNotFoundError
from autorag_research.nodes.reranker.cohere import CohereReranker


class TestCohereRerankerInitialization:
    """Tests for CohereReranker initialization."""

    def test_init_with_api_key_param(self):
        """Test initialization with API key parameter."""
        reranker = CohereReranker(api_key="test-api-key")

        assert reranker.api_key == "test-api-key"
        assert reranker.model == "rerank-v3.5"
        assert reranker.max_concurrency == 10

    def test_init_with_cohere_api_key_env(self, monkeypatch):
        """Test initialization with COHERE_API_KEY env var."""
        monkeypatch.setenv("COHERE_API_KEY", "env-api-key")
        monkeypatch.delenv("CO_API_KEY", raising=False)

        reranker = CohereReranker()

        assert reranker.api_key == "env-api-key"

    def test_init_with_co_api_key_env(self, monkeypatch):
        """Test initialization with CO_API_KEY env var."""
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        monkeypatch.setenv("CO_API_KEY", "co-api-key")

        reranker = CohereReranker()

        assert reranker.api_key == "co-api-key"

    def test_init_cohere_api_key_takes_precedence(self, monkeypatch):
        """Test COHERE_API_KEY takes precedence over CO_API_KEY."""
        monkeypatch.setenv("COHERE_API_KEY", "cohere-key")
        monkeypatch.setenv("CO_API_KEY", "co-key")

        reranker = CohereReranker()

        assert reranker.api_key == "cohere-key"

    def test_init_param_takes_precedence_over_env(self, monkeypatch):
        """Test API key param takes precedence over env vars."""
        monkeypatch.setenv("COHERE_API_KEY", "env-key")

        reranker = CohereReranker(api_key="param-key")

        assert reranker.api_key == "param-key"

    def test_init_no_api_key_raises_error(self, monkeypatch):
        """Test initialization without API key raises error."""
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        monkeypatch.delenv("CO_API_KEY", raising=False)

        with pytest.raises(CohereAPIKeyNotFoundError):
            CohereReranker()

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        reranker = CohereReranker(
            api_key="test-key",
            model="rerank-english-v3.0",
            max_concurrency=5,
        )

        assert reranker.model == "rerank-english-v3.0"
        assert reranker.max_concurrency == 5


def _create_mock_response(results_data: list[tuple[int, float]]) -> MagicMock:
    """Helper to create mock Cohere rerank response.

    Args:
        results_data: List of (index, relevance_score) tuples.

    Returns:
        Mock response object.
    """
    results = []
    for index, score in results_data:
        result = MagicMock()
        result.index = index
        result.relevance_score = score
        results.append(result)

    response = MagicMock()
    response.results = results
    return response


class TestCohereRerankerRun:
    """Tests for CohereReranker run method with mocked client."""

    @pytest.fixture
    def mock_cohere_response(self):
        """Create a mock Cohere rerank response with 3 results."""
        return _create_mock_response([(1, 0.95), (0, 0.75), (2, 0.50)])

    @pytest.fixture
    def reranker(self):
        """Create a CohereReranker instance."""
        return CohereReranker(api_key="test-api-key")

    @pytest.fixture
    def mock_client(self):
        """Create a mock async Cohere client."""
        return AsyncMock()

    def test_run_empty_queries(self, reranker):
        """Test running with empty query list."""
        results = reranker.run(
            queries=[],
            contents_list=[],
            ids_list=[],
            top_k=5,
        )

        assert results == []

    def test_run_empty_contents(self, reranker, mock_client):
        """Test running with empty contents."""
        with patch.object(reranker, "_client", mock_client):
            results = reranker.run(
                queries=["test query"],
                contents_list=[[]],
                ids_list=[[]],
                top_k=5,
            )

            assert len(results) == 1
            assert results[0] == []

    def test_run_single_query(self, reranker, mock_client, mock_cohere_response):
        """Test running with a single query."""
        mock_client.rerank.return_value = mock_cohere_response

        with patch.object(reranker, "_client", mock_client):
            results = reranker.run(
                queries=["What is machine learning?"],
                contents_list=[["Doc A content", "Doc B content", "Doc C content"]],
                ids_list=[[1, 2, 3]],
                top_k=3,
            )

        assert len(results) == 1
        assert len(results[0]) == 3

        # Check first result (highest relevance - Doc B at index 1)
        assert results[0][0]["doc_id"] == 2
        assert results[0][0]["score"] == 0.95
        assert results[0][0]["content"] == "Doc B content"

        # Check second result (Doc A at index 0)
        assert results[0][1]["doc_id"] == 1
        assert results[0][1]["score"] == 0.75

        # Check third result (Doc C at index 2)
        assert results[0][2]["doc_id"] == 3
        assert results[0][2]["score"] == 0.50

        # Verify API was called correctly
        mock_client.rerank.assert_called_once_with(
            model="rerank-v3.5",
            query="What is machine learning?",
            documents=["Doc A content", "Doc B content", "Doc C content"],
            top_n=3,
        )

    def test_run_multiple_queries(self, reranker, mock_client, mock_cohere_response):
        """Test running with multiple queries."""
        mock_client.rerank.return_value = mock_cohere_response

        with patch.object(reranker, "_client", mock_client):
            results = reranker.run(
                queries=["Query 1", "Query 2"],
                contents_list=[
                    ["Doc A", "Doc B", "Doc C"],
                    ["Doc X", "Doc Y", "Doc Z"],
                ],
                ids_list=[[1, 2, 3], [4, 5, 6]],
                top_k=3,
            )

        assert len(results) == 2
        assert len(results[0]) == 3
        assert len(results[1]) == 3
        assert mock_client.rerank.call_count == 2

    def test_run_with_top_k_greater_than_docs(self, reranker, mock_client, mock_cohere_response):
        """Test that top_k is capped at number of documents."""
        mock_client.rerank.return_value = mock_cohere_response

        with patch.object(reranker, "_client", mock_client):
            reranker.run(
                queries=["test"],
                contents_list=[["Doc 1", "Doc 2"]],  # Only 2 docs
                ids_list=[[1, 2]],
                top_k=10,  # Requesting more than available
            )

        # Should call with top_n=2 (min of top_k and len(contents))
        mock_client.rerank.assert_called_once()
        call_args = mock_client.rerank.call_args
        assert call_args.kwargs["top_n"] == 2

    def test_run_with_string_ids(self, reranker, mock_client, mock_cohere_response):
        """Test running with string document IDs."""
        mock_client.rerank.return_value = mock_cohere_response

        with patch.object(reranker, "_client", mock_client):
            results = reranker.run(
                queries=["test query"],
                contents_list=[["Doc A", "Doc B", "Doc C"]],
                ids_list=[["id-a", "id-b", "id-c"]],
                top_k=3,
            )

        # Doc B (index 1) should be first with highest score
        assert results[0][0]["doc_id"] == "id-b"

    def test_run_with_scores_list_ignored(self, reranker, mock_client, mock_cohere_response):
        """Test that scores_list parameter is accepted but ignored."""
        mock_client.rerank.return_value = mock_cohere_response

        with patch.object(reranker, "_client", mock_client):
            results = reranker.run(
                queries=["test"],
                contents_list=[["Doc 1", "Doc 2", "Doc 3"]],
                ids_list=[[1, 2, 3]],
                scores_list=[[0.1, 0.2, 0.3]],  # Provided but should be ignored
                top_k=3,
            )

        assert len(results) == 1

    def test_run_api_error_returns_empty_list(self, reranker, mock_client):
        """Test that API errors result in empty list for that query."""
        mock_client.rerank.side_effect = Exception("API Error")

        with patch.object(reranker, "_client", mock_client):
            results = reranker.run(
                queries=["test"],
                contents_list=[["Doc 1"]],
                ids_list=[[1]],
                top_k=1,
            )

        # Should return empty list on error (not raise)
        assert len(results) == 1
        assert results[0] == []

    def test_client_lazy_initialization(self):
        """Test that client is lazily initialized."""
        reranker = CohereReranker(api_key="test-key")

        # Client should not be initialized yet
        assert reranker._client is None

        # Access client property
        with patch("autorag_research.nodes.reranker.cohere.cohere.AsyncClientV2") as mock_class:
            mock_instance = AsyncMock()
            mock_class.return_value = mock_instance

            client = reranker.client

            # Now client should be initialized
            assert client is mock_instance
            mock_class.assert_called_once_with(api_key="test-key")

            # Second access should return same instance
            client2 = reranker.client
            assert client2 is client
            assert mock_class.call_count == 1  # Not called again


@pytest.mark.api
class TestCohereRerankerAPI:
    """Integration tests for CohereReranker with real API calls.

    These tests require a valid Cohere API key in COHERE_API_KEY env var.
    Run with: pytest -m api
    """

    @pytest.fixture
    def reranker(self):
        """Create a CohereReranker with real API key."""
        api_key = os.getenv("COHERE_API_KEY") or os.getenv("CO_API_KEY")
        if not api_key:
            pytest.skip("COHERE_API_KEY or CO_API_KEY not set")
        return CohereReranker(api_key=api_key)

    def test_real_api_single_query(self, reranker):
        """Test reranking with real API call."""
        results = reranker.run(
            queries=["What is machine learning?"],
            contents_list=[
                [
                    "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
                    "The weather today is sunny with a high of 75 degrees.",
                    "Deep learning uses neural networks with many layers to process complex patterns.",
                ]
            ],
            ids_list=[[1, 2, 3]],
            top_k=3,
        )

        assert len(results) == 1
        assert len(results[0]) == 3

        # All results should have required keys
        for result in results[0]:
            assert "doc_id" in result
            assert "score" in result
            assert "content" in result
            assert isinstance(result["score"], float)
            assert 0 <= result["score"] <= 1

        # The ML-related docs should rank higher than the weather doc
        doc_ids_in_order = [r["doc_id"] for r in results[0]]
        weather_doc_index = doc_ids_in_order.index(2)
        # Weather doc should not be first (ML docs should rank higher)
        assert weather_doc_index > 0

    def test_real_api_multiple_queries(self, reranker):
        """Test reranking multiple queries with real API call."""
        results = reranker.run(
            queries=["Capital cities", "Programming languages"],
            contents_list=[
                ["Paris is the capital of France", "Python is a programming language", "Tokyo is the capital of Japan"],
                ["Java is used for enterprise applications", "Berlin is in Germany", "JavaScript runs in browsers"],
            ],
            ids_list=[[1, 2, 3], [4, 5, 6]],
            top_k=2,
        )

        assert len(results) == 2
        assert len(results[0]) == 2
        assert len(results[1]) == 2
