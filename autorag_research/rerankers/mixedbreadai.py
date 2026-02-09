"""Mixedbread AI reranker implementation."""

from __future__ import annotations

import os

import httpx
from pydantic import Field

from autorag_research.rerankers.base import BaseReranker, RerankResult

MIXEDBREAD_RERANK_URL = "https://api.mixedbread.ai/v1/reranking"


class MixedbreadAIReranker(BaseReranker):
    """Reranker using Mixedbread AI's rerank API.

    Requires `MIXEDBREAD_API_KEY` environment variable.
    Uses httpx for API calls.
    """

    model_name: str = Field(default="mixedbread-ai/mxbai-rerank-large-v1", description="Mixedbread AI rerank model.")
    api_key: str | None = Field(
        default=None, exclude=True, description="Mixedbread API key. If None, uses MIXEDBREAD_API_KEY env var."
    )

    _api_key: str | None = None
    _client: httpx.Client | None = None
    _async_client: httpx.AsyncClient | None = None

    def model_post_init(self, __context) -> None:
        """Initialize API key and HTTP clients after model creation."""
        self._api_key = self.api_key or os.environ.get("MIXEDBREAD_API_KEY")
        if not self._api_key:
            msg = "MIXEDBREAD_API_KEY environment variable is not set"
            raise ValueError(msg)

        # Initialize reusable HTTP clients
        self._client = httpx.Client(timeout=60.0)
        self._async_client = httpx.AsyncClient(timeout=60.0)

    def __del__(self) -> None:
        """Cleanup HTTP clients on deletion."""
        if self._client is not None:
            self._client.close()
        if self._async_client is not None:
            # AsyncClient cleanup is best-effort in __del__
            try:
                import asyncio

                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._async_client.aclose())  # noqa: RUF006
                else:
                    loop.run_until_complete(self._async_client.aclose())
            except Exception:  # noqa: S110
                pass  # Best effort cleanup

    def _get_headers(self) -> dict[str, str]:
        """Get headers for Mixedbread API requests."""
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _parse_response(self, response_data: dict, documents: list[str]) -> list[RerankResult]:
        """Parse Mixedbread API response into RerankResult objects."""
        results = response_data.get("data", [])
        return [
            RerankResult(
                index=result["index"],
                text=documents[result["index"]],
                score=result["score"],
            )
            for result in results
        ]

    def rerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        """Rerank documents using Mixedbread AI's rerank API.

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
            "input": documents,
            "top_k": top_k,
            "return_input": False,
        }

        response = self._client.post(
            MIXEDBREAD_RERANK_URL,
            headers=self._get_headers(),
            json=payload,
        )
        response.raise_for_status()
        return self._parse_response(response.json(), documents)

    async def arerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        """Rerank documents asynchronously using Mixedbread AI's rerank API.

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
            "input": documents,
            "top_k": top_k,
            "return_input": False,
        }

        response = await self._async_client.post(
            MIXEDBREAD_RERANK_URL,
            headers=self._get_headers(),
            json=payload,
        )
        response.raise_for_status()
        return self._parse_response(response.json(), documents)
