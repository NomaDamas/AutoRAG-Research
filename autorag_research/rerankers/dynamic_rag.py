"""DynamicRAG inference-time reranker.

DynamicRAG dynamically reorders and truncates retrieved candidates before
answer generation. This implementation provides an AutoRAG-compatible inference
reranker: it can wrap an existing reranker for scoring, then applies a dynamic
cut policy based on minimum score, score gap, and min/max retained documents.
"""

from __future__ import annotations

from pydantic import Field

from autorag_research.rerankers.base import BaseReranker, RerankResult


class DynamicRAGReranker(BaseReranker):
    """Reranker with DynamicRAG-style variable top-k truncation."""

    model_name: str = Field(default="dynamic-rag", description="DynamicRAG reranker wrapper name.")
    base_reranker: BaseReranker | None = Field(default=None, description="Optional underlying scorer/reranker.")
    min_top_k: int = Field(default=1, ge=1, description="Minimum documents to keep.")
    max_top_k: int | None = Field(default=None, description="Maximum documents to keep before requested top_k.")
    score_drop_threshold: float | None = Field(
        default=0.25,
        description="Stop after min_top_k when adjacent score drop exceeds this threshold.",
    )
    min_score: float | None = Field(default=None, description="Stop after min_top_k when score falls below this value.")

    def _score_documents(self, query: str, documents: list[str]) -> list[RerankResult]:
        """Score documents with base reranker or deterministic order fallback."""
        if self.base_reranker is not None:
            return self.base_reranker.rerank(query, documents, top_k=len(documents))

        return [
            RerankResult(index=index, text=document, score=float(len(documents) - index))
            for index, document in enumerate(documents)
        ]

    def _dynamic_cut(
        self, ranked_results: list[RerankResult], requested_top_k: int | None, *, allow_score_cut: bool
    ) -> int:
        """Choose an effective cutoff from ranked results."""
        if not ranked_results:
            return 0

        upper_bound = len(ranked_results)
        if requested_top_k is not None:
            upper_bound = min(upper_bound, requested_top_k)
        if self.max_top_k is not None:
            upper_bound = min(upper_bound, self.max_top_k)
        upper_bound = max(1, upper_bound)
        min_keep = min(self.min_top_k, upper_bound)

        effective_k = upper_bound
        if not allow_score_cut:
            return effective_k

        for position in range(min_keep, upper_bound):
            current_score = ranked_results[position].score
            previous_score = ranked_results[position - 1].score
            if self.min_score is not None and current_score < self.min_score:
                effective_k = position
                break
            if self.score_drop_threshold is not None and previous_score - current_score >= self.score_drop_threshold:
                effective_k = position
                break

        return max(min_keep, effective_k)

    def rerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        """Rerank documents and return a dynamically truncated result list."""
        if not documents:
            return []
        ranked_results = self._score_documents(query, documents)
        ranked_results.sort(key=lambda result: result.score, reverse=True)
        effective_k = self._dynamic_cut(ranked_results, top_k, allow_score_cut=self.base_reranker is not None)
        return ranked_results[:effective_k]

    async def arerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        """Async rerank implementation."""
        if not documents:
            return []
        if self.base_reranker is not None:
            ranked_results = await self.base_reranker.arerank(query, documents, top_k=len(documents))
            ranked_results.sort(key=lambda result: result.score, reverse=True)
            effective_k = self._dynamic_cut(ranked_results, top_k, allow_score_cut=self.base_reranker is not None)
            return ranked_results[:effective_k]
        return self.rerank(query, documents, top_k)


__all__ = ["DynamicRAGReranker"]
