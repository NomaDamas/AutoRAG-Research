"""Mixedbread AI reranker implementation."""

from __future__ import annotations

import os

from pydantic import Field

from autorag_research.rerankers.api_base import APIReranker
from autorag_research.rerankers.base import RerankResult

MIXEDBREAD_RERANK_URL = "https://api.mixedbread.ai/v1/reranking"


class MixedbreadAIReranker(APIReranker):
    """Reranker using Mixedbread AI's rerank API.

    Requires `MIXEDBREAD_API_KEY` environment variable.
    Includes automatic retry with exponential backoff for transient errors.
    """

    model_name: str = Field(default="mixedbread-ai/mxbai-rerank-large-v1", description="Mixedbread AI rerank model.")
    api_key: str | None = Field(
        default=None, exclude=True, description="Mixedbread API key. If None, uses MIXEDBREAD_API_KEY env var."
    )

    _api_key: str | None = None

    def model_post_init(self, __context) -> None:
        """Initialize API key and HTTP clients after model creation."""
        self._api_key = self.api_key or os.environ.get("MIXEDBREAD_API_KEY")
        if not self._api_key:
            msg = "MIXEDBREAD_API_KEY environment variable is not set"
            raise ValueError(msg)

        # Initialize HTTP clients from parent
        super().model_post_init(__context)

    def _get_api_url(self) -> str:
        """Get the API endpoint URL."""
        return MIXEDBREAD_RERANK_URL

    def _get_headers(self) -> dict[str, str]:
        """Get headers for Mixedbread API requests."""
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(self, query: str, documents: list[str], top_k: int) -> dict:
        """Build the API request payload."""
        return {
            "model": self.model_name,
            "query": query,
            "input": documents,
            "top_k": top_k,
            "return_input": False,
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
