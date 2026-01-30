"""Test cases for CohereReranker."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autorag_research.exceptions import CohereAPIKeyNotFoundError
from autorag_research.nodes.reranker.cohere import CohereReranker


class TestCohereReranker:
    """Tests for CohereReranker."""

    def test_init_sets_defaults(self):
        """Test initialization with default values."""
        reranker = CohereReranker(api_key="test-api-key")

        assert reranker.api_key == "test-api-key"
        assert reranker.model == "rerank-v4.0-fast"
        assert reranker.max_tokens_per_doc is None
        assert reranker.max_concurrency == 10

    def test_init_raises_error_without_api_key(self, monkeypatch):
        """Test that missing API key raises error."""
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        monkeypatch.delenv("CO_API_KEY", raising=False)

        with pytest.raises(CohereAPIKeyNotFoundError):
            CohereReranker()

    def test_run_reranks_documents(self):
        """Test that run() reranks documents correctly."""
        reranker = CohereReranker(api_key="test-api-key")

        # Create mock response
        mock_results = []
        for idx, score in [(1, 0.95), (0, 0.75), (2, 0.50)]:
            r = MagicMock()
            r.index = idx
            r.relevance_score = score
            mock_results.append(r)

        mock_response = MagicMock()
        mock_response.results = mock_results

        mock_client = AsyncMock()
        mock_client.rerank.return_value = mock_response

        with patch.object(reranker, "_client", mock_client):
            results = reranker.run(
                queries=["What is ML?"],
                contents_list=[["Doc A", "Doc B", "Doc C"]],
                ids_list=[[1, 2, 3]],
                top_k=3,
            )

        assert len(results) == 1
        assert len(results[0]) == 3
        # Doc B (index 1) ranked first with highest score
        assert results[0][0] == {"doc_id": 2, "score": 0.95, "content": "Doc B"}
        assert results[0][1] == {"doc_id": 1, "score": 0.75, "content": "Doc A"}
        assert results[0][2] == {"doc_id": 3, "score": 0.50, "content": "Doc C"}

    def test_run_handles_api_error(self):
        """Test that API errors return empty list instead of raising."""
        reranker = CohereReranker(api_key="test-api-key")
        mock_client = AsyncMock()
        mock_client.rerank.side_effect = Exception("API Error")

        with patch.object(reranker, "_client", mock_client):
            results = reranker.run(
                queries=["test"],
                contents_list=[["Doc 1"]],
                ids_list=[[1]],
                top_k=1,
            )

        assert results == [[]]


@pytest.mark.api
class TestCohereRerankerAPI:
    """Integration test with real Cohere API."""

    def test_real_api_call(self):
        """Test reranking with real API."""
        api_key = os.getenv("COHERE_API_KEY") or os.getenv("CO_API_KEY")
        if not api_key:
            pytest.skip("COHERE_API_KEY not set")

        reranker = CohereReranker(api_key=api_key)
        results = reranker.run(
            queries=["What is machine learning?"],
            contents_list=[
                [
                    "Machine learning enables systems to learn from data.",
                    "The weather today is sunny.",
                    "Deep learning uses neural networks.",
                ]
            ],
            ids_list=[[1, 2, 3]],
            top_k=3,
        )

        assert len(results) == 1
        assert len(results[0]) == 3
        for r in results[0]:
            assert "doc_id" in r
            assert "score" in r
            assert "content" in r
            assert 0 <= r["score"] <= 1
