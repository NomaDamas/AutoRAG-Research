"""Cohere reranker implementation."""

from __future__ import annotations

import os
from typing import Any

from pydantic import Field

from autorag_research.rerankers.api_base import APIReranker, _create_retry_decorator
from autorag_research.rerankers.base import RerankResult


class CohereReranker(APIReranker):
    """Reranker using Cohere's rerank API.

    Requires the `cohere` package: `pip install cohere`
    Requires `COHERE_API_KEY` environment variable.
    Includes automatic retry with exponential backoff for transient errors.
    """

    model_name: str = Field(default="rerank-v3.5", description="Cohere rerank model name.")
    api_key: str | None = Field(
        default=None, exclude=True, description="Cohere API key. If None, uses COHERE_API_KEY env var."
    )

    _client: Any = None
    _async_client: Any = None

    def model_post_init(self, __context) -> None:
        """Initialize Cohere clients after model creation."""
        try:
            import cohere
        except ImportError as e:
            msg = "cohere package is required. Install with: pip install cohere"
            raise ImportError(msg) from e

        api_key = self.api_key or os.environ.get("COHERE_API_KEY")
        if not api_key:
            msg = "COHERE_API_KEY environment variable is not set"
            raise ValueError(msg)

        self._client = cohere.Client(api_key=api_key)
        self._async_client = cohere.AsyncClient(api_key=api_key)

    def rerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        """Rerank documents using Cohere's rerank API with automatic retry.

        Args:
            query: The search query.
            documents: List of document texts to rerank.
            top_k: Number of top results to return. If None, returns all documents.

        Returns:
            List of RerankResult objects sorted by relevance score (descending).
        """
        if not documents:
            return []

        top_k = top_k or len(documents)

        @_create_retry_decorator()
        def _call_api():
            return self._client.rerank(
                model=self.model_name,
                query=query,
                documents=documents,
                top_n=top_k,
            )

        response = _call_api()

        return [
            RerankResult(
                index=result.index,
                text=documents[result.index],
                score=result.relevance_score,
            )
            for result in response.results
        ]

    async def arerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        """Rerank documents asynchronously using Cohere's rerank API with automatic retry.

        Args:
            query: The search query.
            documents: List of document texts to rerank.
            top_k: Number of top results to return. If None, returns all documents.

        Returns:
            List of RerankResult objects sorted by relevance score (descending).
        """
        if not documents:
            return []

        top_k = top_k or len(documents)

        @_create_retry_decorator()
        async def _call_api():
            return await self._async_client.rerank(
                model=self.model_name,
                query=query,
                documents=documents,
                top_n=top_k,
            )

        response = await _call_api()

        return [
            RerankResult(
                index=result.index,
                text=documents[result.index],
                score=result.relevance_score,
            )
            for result in response.results
        ]
