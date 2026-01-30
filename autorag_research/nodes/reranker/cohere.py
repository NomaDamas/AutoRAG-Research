"""
Cohere Reranker module using Cohere Rerank API.

This module implements reranking using Cohere's Rerank API,
enabling re-scoring of retrieved documents based on query relevance.
"""

import asyncio
import os
from typing import Any

import cohere

from autorag_research.exceptions import CohereAPIKeyNotFoundError
from autorag_research.nodes import BaseModule
from autorag_research.util import run_with_concurrency_limit


class CohereReranker(BaseModule):
    """
    Reranker module using Cohere Rerank API.

    This module reranks documents using Cohere's neural reranking model,
    improving retrieval quality by re-scoring documents based on semantic
    relevance to the query.

    Attributes:
        model: Cohere rerank model name (default: "rerank-v3.5").
        max_concurrency: Maximum concurrent API calls (default: 10).

    Example:
        ```python
        reranker = CohereReranker(api_key="your-api-key")
        results = reranker.run(
            queries=["What is machine learning?"],
            contents_list=[["ML is...", "Deep learning is...", "Python is..."]],
            ids_list=[[1, 2, 3]],
            top_k=2,
        )
        ```
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "rerank-v3.5",
        max_concurrency: int = 10,
    ):
        """
        Initialize Cohere Reranker.

        Args:
            api_key: Cohere API key. If not provided, will try to get from
                COHERE_API_KEY or CO_API_KEY environment variables.
            model: Cohere rerank model name (default: "rerank-v3.5").
            max_concurrency: Maximum number of concurrent API calls (default: 10).

        Raises:
            CohereAPIKeyNotFoundError: If no API key is found.
        """
        self.api_key = api_key or os.getenv("COHERE_API_KEY") or os.getenv("CO_API_KEY")
        if not self.api_key:
            raise CohereAPIKeyNotFoundError()

        self.model = model
        self.max_concurrency = max_concurrency
        self._client: cohere.AsyncClientV2 | None = None

    def run(
        self,
        queries: list[str],
        contents_list: list[list[str]],
        ids_list: list[list[int | str]],
        scores_list: list[list[float]] | None = None,
        top_k: int = 10,
    ) -> list[list[dict[str, Any]]]:
        """
        Rerank documents for given queries using Cohere Rerank API.

        Args:
            queries: List of query strings.
            contents_list: List of document content lists for each query.
            ids_list: List of document ID lists corresponding to contents_list.
            scores_list: Optional list of original scores (not used in reranking).
            top_k: Number of top documents to return per query.

        Returns:
            List of reranked results for each query. Each result is a list of
            dictionaries containing:
            - doc_id: Document ID (from ids_list)
            - score: Cohere relevance score (0-1, higher = more relevant)
            - content: Document content
        """
        return asyncio.run(self._run_async(queries, contents_list, ids_list, top_k))

    @property
    def client(self) -> cohere.AsyncClientV2:
        """Lazy-initialized async Cohere client."""
        if self._client is None:
            self._client = cohere.AsyncClientV2(api_key=self.api_key)
        return self._client

    async def _run_async(
        self,
        queries: list[str],
        contents_list: list[list[str]],
        ids_list: list[list[int | str]],
        top_k: int,
    ) -> list[list[dict[str, Any]]]:
        """Async implementation of reranking."""

        # Prepare rerank tasks
        tasks_data = list(zip(queries, contents_list, ids_list, strict=True))

        async def rerank_single(
            task_data: tuple[str, list[str], list[int | str]],
        ) -> list[dict[str, Any]]:
            query, contents, ids = task_data

            if not contents:
                return []

            # Call Cohere rerank API
            response = await self.client.rerank(
                model=self.model,
                query=query,
                documents=contents,
                top_n=min(top_k, len(contents)),
            )

            # Build results from reranked documents
            results = []
            for result in response.results:
                idx = result.index
                results.append({
                    "doc_id": ids[idx],
                    "score": result.relevance_score,
                    "content": contents[idx],
                })

            return results

        # Run with concurrency limit
        results = await run_with_concurrency_limit(
            items=tasks_data,
            async_func=rerank_single,
            max_concurrency=self.max_concurrency,
            error_message="Cohere rerank API call failed",
        )

        # Replace None results (from failed tasks) with empty lists
        return [r if r is not None else [] for r in results]
