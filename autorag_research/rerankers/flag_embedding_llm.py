"""FlagEmbedding LLM-based reranker implementation."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pydantic import ConfigDict, Field

from autorag_research.rerankers.base import BaseReranker, RerankResult

logger = logging.getLogger("AutoRAG-Research")


class FlagEmbeddingLLMReranker(BaseReranker):
    """Reranker using FlagEmbedding's FlagLLMReranker.

    Uses LLM-based reranking models (e.g., BAAI/bge-reranker-v2-gemma)
    for higher quality reranking at the cost of more compute.

    Requires the `FlagEmbedding` package: `pip install FlagEmbedding`
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = Field(
        default="BAAI/bge-reranker-v2-gemma",
        description="FlagEmbedding LLM reranker model name.",
    )
    use_fp16: bool = Field(default=False, description="Use FP16 for inference.")

    _model: Any = None

    def model_post_init(self, __context) -> None:
        """Initialize FlagLLMReranker model after creation."""
        try:
            from FlagEmbedding import FlagLLMReranker  # ty: ignore[unresolved-import]
        except ImportError as e:
            msg = "FlagEmbedding package is required. Install with: pip install FlagEmbedding"
            raise ImportError(msg) from e

        self._model = FlagLLMReranker(self.model_name, use_fp16=self.use_fp16)
        logger.info("Loaded FlagEmbedding LLM reranker: %s", self.model_name)

    def rerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        """Rerank documents using FlagLLMReranker scoring.

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

        pairs = [[query, doc] for doc in documents]
        scores = self._model.compute_score(pairs)

        # compute_score returns a single float for one pair, list for multiple
        if isinstance(scores, float):
            scores = [scores]

        results = [
            RerankResult(index=i, text=doc, score=float(score))
            for i, (doc, score) in enumerate(zip(documents, scores, strict=True))
        ]
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    async def arerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        """Rerank documents asynchronously using FlagLLMReranker scoring.

        Args:
            query: The search query.
            documents: List of document texts to rerank.
            top_k: Number of top results to return. If None, returns all documents.

        Returns:
            List of RerankResult objects sorted by relevance score (descending).
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.rerank, query, documents, top_k)
