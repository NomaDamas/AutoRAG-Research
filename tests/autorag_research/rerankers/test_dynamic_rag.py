"""Tests for DynamicRAG reranker."""

from autorag_research.rerankers.base import BaseReranker, RerankResult
from autorag_research.rerankers.dynamic_rag import DynamicRAGReranker


class FixedReranker(BaseReranker):
    """Fixed-score reranker for tests."""

    scores: list[float]

    def rerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        results = [RerankResult(index=index, text=document, score=self.scores[index]) for index, document in enumerate(documents)]
        results.sort(key=lambda result: result.score, reverse=True)
        return results[:top_k] if top_k is not None else results

    async def arerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        return self.rerank(query, documents, top_k)


def test_dynamic_rag_reranker_cuts_on_score_drop():
    reranker = DynamicRAGReranker(
        base_reranker=FixedReranker(scores=[0.95, 0.9, 0.4, 0.39]),
        min_top_k=2,
        score_drop_threshold=0.25,
    )

    results = reranker.rerank("query", ["a", "b", "c", "d"], top_k=4)

    assert [result.text for result in results] == ["a", "b"]


def test_dynamic_rag_reranker_respects_min_top_k():
    reranker = DynamicRAGReranker(
        base_reranker=FixedReranker(scores=[0.9, 0.1, 0.0]),
        min_top_k=2,
        score_drop_threshold=0.2,
        min_score=0.2,
    )

    results = reranker.rerank("query", ["a", "b", "c"], top_k=3)

    assert [result.text for result in results] == ["a", "b"]


def test_dynamic_rag_reranker_fallback_preserves_order_scores():
    reranker = DynamicRAGReranker(min_top_k=1, score_drop_threshold=None, max_top_k=2)

    results = reranker.rerank("query", ["a", "b", "c"], top_k=3)

    assert [(result.index, result.text, result.score) for result in results] == [(0, "a", 3.0), (1, "b", 2.0)]
