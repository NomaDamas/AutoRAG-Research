"""Voyage AI reranker implementation."""

from __future__ import annotations

import os
from typing import Any

from pydantic import Field

from autorag_research.rerankers.base import BaseReranker, RerankResult


class VoyageAIReranker(BaseReranker):
    """Reranker using Voyage AI's rerank API.

    Requires the `voyageai` package: `pip install voyageai`
    Requires `VOYAGE_API_KEY` environment variable.
    """

    model_name: str = Field(default="rerank-2", description="Voyage AI rerank model name.")
    api_key: str | None = Field(
        default=None, exclude=True, description="Voyage API key. If None, uses VOYAGE_API_KEY env var."
    )

    _client: Any = None
    _async_client: Any = None

    def model_post_init(self, __context) -> None:
        """Initialize Voyage AI clients after model creation."""
        try:
            import voyageai  # ty: ignore[unresolved-import]
        except ImportError as e:
            msg = "voyageai package is required. Install with: pip install voyageai"
            raise ImportError(msg) from e

        api_key = self.api_key or os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            msg = "VOYAGE_API_KEY environment variable is not set"
            raise ValueError(msg)

        self._client = voyageai.Client(api_key=api_key)
        self._async_client = voyageai.AsyncClient(api_key=api_key)

    def rerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        """Rerank documents using Voyage AI's rerank API.

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

        response = self._client.rerank(
            model=self.model_name,
            query=query,
            documents=documents,
            top_k=top_k,
        )

        return [
            RerankResult(
                index=result.index,
                text=documents[result.index],
                score=result.relevance_score,
            )
            for result in response.results
        ]

    async def arerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        """Rerank documents asynchronously using Voyage AI's rerank API.

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

        response = await self._async_client.rerank(
            model=self.model_name,
            query=query,
            documents=documents,
            top_k=top_k,
        )

        return [
            RerankResult(
                index=result.index,
                text=documents[result.index],
                score=result.relevance_score,
            )
            for result in response.results
        ]
