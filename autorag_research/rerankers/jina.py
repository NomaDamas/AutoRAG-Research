"""Jina reranker implementation."""

from __future__ import annotations

import os

from pydantic import Field

from autorag_research.rerankers.api_base import APIReranker
from autorag_research.rerankers.base import RerankResult

JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"


class JinaReranker(APIReranker):
    """Reranker using Jina AI's rerank API.

    Requires `JINA_API_KEY` environment variable.
    Includes automatic retry with exponential backoff for transient errors.
    """

    model_name: str = Field(default="jina-reranker-v2-base-multilingual", description="Jina rerank model name.")
    api_key: str | None = Field(
        default=None, exclude=True, description="Jina API key. If None, uses JINA_API_KEY env var."
    )

    _api_key: str | None = None

    def model_post_init(self, __context) -> None:
        """Initialize API key and HTTP clients after model creation."""
        self._api_key = self.api_key or os.environ.get("JINA_API_KEY")
        if not self._api_key:
            msg = "JINA_API_KEY environment variable is not set"
            raise ValueError(msg)

        # Initialize HTTP clients from parent
        super().model_post_init(__context)

    def _get_api_url(self) -> str:
        """Get the API endpoint URL."""
        return JINA_RERANK_URL

    def _get_headers(self) -> dict[str, str]:
        """Get headers for Jina API requests."""
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(self, query: str, documents: list[str], top_k: int) -> dict:
        """Build the API request payload."""
        return {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "top_n": top_k,
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
