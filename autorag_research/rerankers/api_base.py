"""Base class for API-based rerankers with retry logic."""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING

from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from autorag_research.rerankers.base import BaseReranker, RerankResult

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger("AutoRAG-Research")


def _create_retry_decorator():
    """Create a retry decorator for API calls with exponential backoff."""
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


class HTTPAPIReranker(BaseReranker):
    """Base class for HTTP-based API rerankers using httpx.

    Provides:
    - Reusable httpx.Client and httpx.AsyncClient with proper cleanup
    - Automatic retry with exponential backoff for transient errors
    - Common patterns for API calls

    Subclasses must implement:
    - _get_api_url() -> str
    - _get_headers() -> dict[str, str]
    - _build_payload(query, documents, top_k) -> dict
    - _parse_response(response_data, documents) -> list[RerankResult]
    """

    _client: httpx.Client | None = None
    _async_client: httpx.AsyncClient | None = None

    def model_post_init(self, __context) -> None:
        """Initialize HTTP clients after model creation."""
        import httpx

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

    @abstractmethod
    def _get_api_url(self) -> str:
        """Get the API endpoint URL."""
        ...

    @abstractmethod
    def _get_headers(self) -> dict[str, str]:
        """Get headers for API requests."""
        ...

    @abstractmethod
    def _build_payload(self, query: str, documents: list[str], top_k: int) -> dict:
        """Build the API request payload."""
        ...

    @abstractmethod
    def _parse_response(self, response_data: dict, documents: list[str]) -> list[RerankResult]:
        """Parse API response into RerankResult objects."""
        ...

    def rerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        """Rerank documents using the API with automatic retry.

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
            response = self._client.post(
                self._get_api_url(),
                headers=self._get_headers(),
                json=self._build_payload(query, documents, top_k),
            )
            response.raise_for_status()
            return response.json()

        response_data = _call_api()
        return self._parse_response(response_data, documents)

    async def arerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        """Rerank documents asynchronously using the API with automatic retry.

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
            response = await self._async_client.post(
                self._get_api_url(),
                headers=self._get_headers(),
                json=self._build_payload(query, documents, top_k),
            )
            response.raise_for_status()
            return response.json()

        response_data = await _call_api()
        return self._parse_response(response_data, documents)
