"""Base classes for rerankers."""

from __future__ import annotations

import asyncio
from abc import abstractmethod
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field


@dataclass
class RerankResult:
    """Single reranked document result."""

    index: int  # Original index in input documents
    text: str  # Document text
    score: float  # Relevance score


class BaseReranker(BaseModel):
    """Base class for rerankers following MultiVectorBaseEmbedding pattern.

    Unlike simple embedding retrieval, rerankers score query-document pairs
    to provide more accurate relevance rankings.

    Uses consistent method naming:
    - rerank: Rerank documents for a single query
    - rerank_documents: Rerank for multiple queries
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = Field(default="unknown", description="The reranker model name.")
    batch_size: int = Field(default=64, description="Batch size for reranking.")

    @abstractmethod
    def rerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        """Rerank documents for a single query.

        Args:
            query: The search query.
            documents: List of document texts to rerank.
            top_k: Number of top results to return. If None, returns all documents.

        Returns:
            List of RerankResult objects sorted by relevance score (descending).
        """

    @abstractmethod
    async def arerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        """Rerank documents asynchronously for a single query.

        Args:
            query: The search query.
            documents: List of document texts to rerank.
            top_k: Number of top results to return. If None, returns all documents.

        Returns:
            List of RerankResult objects sorted by relevance score (descending).
        """

    def rerank_documents(
        self,
        queries: list[str],
        documents_list: list[list[str]],
        top_k: int | None = None,
    ) -> list[list[RerankResult]]:
        """Rerank documents for multiple queries.

        Args:
            queries: List of search queries.
            documents_list: List of document lists, one per query.
            top_k: Number of top results to return per query. If None, returns all.

        Returns:
            List of RerankResult lists, one per query.
        """
        return [self.rerank(q, docs, top_k) for q, docs in zip(queries, documents_list, strict=True)]

    async def arerank_documents(
        self,
        queries: list[str],
        documents_list: list[list[str]],
        top_k: int | None = None,
    ) -> list[list[RerankResult]]:
        """Rerank documents asynchronously for multiple queries.

        Args:
            queries: List of search queries.
            documents_list: List of document lists, one per query.
            top_k: Number of top results to return per query. If None, returns all.

        Returns:
            List of RerankResult lists, one per query.
        """
        return list(
            await asyncio.gather(*[
                self.arerank(q, docs, top_k) for q, docs in zip(queries, documents_list, strict=True)
            ])
        )

    def rerank_documents_batch(
        self,
        queries: list[str],
        documents_list: list[list[str]],
        top_k: int | None = None,
    ) -> list[list[RerankResult]]:
        """Batched reranking with self.batch_size.

        Args:
            queries: List of search queries.
            documents_list: List of document lists, one per query.
            top_k: Number of top results to return per query. If None, returns all.

        Returns:
            List of RerankResult lists, one per query.
        """
        results: list[list[RerankResult]] = []
        for i in range(0, len(queries), self.batch_size):
            batch_queries = queries[i : i + self.batch_size]
            batch_docs = documents_list[i : i + self.batch_size]
            results.extend(self.rerank_documents(batch_queries, batch_docs, top_k))
        return results

    async def arerank_documents_batch(
        self,
        queries: list[str],
        documents_list: list[list[str]],
        top_k: int | None = None,
    ) -> list[list[RerankResult]]:
        """Batched async reranking with self.batch_size.

        Args:
            queries: List of search queries.
            documents_list: List of document lists, one per query.
            top_k: Number of top results to return per query. If None, returns all.

        Returns:
            List of RerankResult lists, one per query.
        """
        results: list[list[RerankResult]] = []
        for i in range(0, len(queries), self.batch_size):
            batch_queries = queries[i : i + self.batch_size]
            batch_docs = documents_list[i : i + self.batch_size]
            batch_results = await self.arerank_documents(batch_queries, batch_docs, top_k)
            results.extend(batch_results)
        return results
