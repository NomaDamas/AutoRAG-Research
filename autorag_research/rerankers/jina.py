"""Jina reranker implementation."""

from __future__ import annotations

import os

import httpx
from pydantic import Field

from autorag_research.rerankers.base import BaseReranker, RerankResult

JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"


class JinaReranker(BaseReranker):
    """Reranker using Jina AI's rerank API.

    Requires `JINA_API_KEY` environment variable.
    Uses httpx for API calls (already in project dependencies).
    """

    model_name: str = Field(default="jina-reranker-v2-base-multilingual", description="Jina rerank model name.")
    api_key: str | None = Field(
        default=None, exclude=True, description="Jina API key. If None, uses JINA_API_KEY env var."
    )

    _api_key: str | None = None

    def model_post_init(self, __context) -> None:
        """Initialize API key after model creation."""
        self._api_key = self.api_key or os.environ.get("JINA_API_KEY")
        if not self._api_key:
            msg = "JINA_API_KEY environment variable is not set"
            raise ValueError(msg)

    def _get_headers(self) -> dict[str, str]:
        """Get headers for Jina API requests."""
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _parse_response(self, response_data: dict, documents: list[str]) -> list[RerankResult]:
        """Parse Jina API response into RerankResult objects."""
        results = response_data.get("results", [])
        return [
            RerankResult(
                index=result["index"],
                text=documents[result["index"]],
                score=result["relevance_score"],
            )
            for result in results
        ]

    def rerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        """Rerank documents using Jina's rerank API.

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

        payload = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "top_n": top_k,
        }

        with httpx.Client() as client:
            response = client.post(
                JINA_RERANK_URL,
                headers=self._get_headers(),
                json=payload,
                timeout=60.0,
            )
            response.raise_for_status()
            return self._parse_response(response.json(), documents)

    async def arerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        """Rerank documents asynchronously using Jina's rerank API.

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

        payload = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "top_n": top_k,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                JINA_RERANK_URL,
                headers=self._get_headers(),
                json=payload,
                timeout=60.0,
            )
            response.raise_for_status()
            return self._parse_response(response.json(), documents)
