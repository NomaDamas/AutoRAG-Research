"""FlashRank reranker implementation."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pydantic import ConfigDict, Field

from autorag_research.rerankers.base import BaseReranker, RerankResult

logger = logging.getLogger("AutoRAG-Research")


class FlashRankReranker(BaseReranker):
    """Reranker using FlashRank's lightweight ONNX-based models.

    FlashRank provides fast, CPU-friendly reranking without GPU requirements.

    Requires the `flashrank` package: `pip install flashrank`
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = Field(
        default="ms-marco-MiniLM-L-12-v2",
        description="FlashRank model name.",
    )
    max_length: int = Field(default=512, description="Maximum input sequence length.")

    _ranker: Any = None

    def model_post_init(self, __context) -> None:
        """Initialize FlashRank Ranker after creation."""
        try:
            from flashrank import Ranker  # ty: ignore[unresolved-import]
        except ImportError as e:
            msg = "flashrank package is required. Install with: pip install flashrank"
            raise ImportError(msg) from e

        self._ranker = Ranker(model_name=self.model_name, max_length=self.max_length)
        logger.info("Loaded FlashRank reranker: %s", self.model_name)

    def rerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        """Rerank documents using FlashRank scoring.

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
        top_k = min(top_k, len(documents))

        try:
            from flashrank import RerankRequest  # ty: ignore[unresolved-import]
        except ImportError as e:
            msg = "flashrank package is required. Install with: pip install flashrank"
            raise ImportError(msg) from e

        passages = [{"id": i, "text": doc} for i, doc in enumerate(documents)]
        request = RerankRequest(query=query, passages=passages)
        response = self._ranker.rerank(request)

        results = [RerankResult(index=int(r["id"]), text=r["text"], score=float(r["score"])) for r in response]
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    async def arerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        """Rerank documents asynchronously using FlashRank scoring.

        Args:
            query: The search query.
            documents: List of document texts to rerank.
            top_k: Number of top results to return. If None, returns all documents.

        Returns:
            List of RerankResult objects sorted by relevance score (descending).
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.rerank, query, documents, top_k)
