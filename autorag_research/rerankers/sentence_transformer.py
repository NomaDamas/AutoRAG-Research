"""SentenceTransformer CrossEncoder reranker implementation."""

from __future__ import annotations

import logging

from pydantic import Field

from autorag_research.rerankers.base import RerankResult
from autorag_research.rerankers.local_base import LocalReranker

logger = logging.getLogger("AutoRAG-Research")


class SentenceTransformerReranker(LocalReranker):
    """Reranker using SentenceTransformers CrossEncoder models.

    Uses cross-encoder models to score query-document pairs for reranking.

    Requires the `sentence-transformers` package: `pip install sentence-transformers`
    """

    model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-2-v2",
        description="CrossEncoder model name from HuggingFace.",
    )

    def model_post_init(self, __context) -> None:
        """Initialize CrossEncoder model after creation."""
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            msg = "sentence-transformers package is required. Install with: pip install sentence-transformers"
            raise ImportError(msg) from e

        self._init_device()
        self._model = CrossEncoder(self.model_name, max_length=self.max_length, device=self._device)
        logger.info("Loaded SentenceTransformer CrossEncoder: %s on %s", self.model_name, self._device)

    def rerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        """Rerank documents using CrossEncoder scoring.

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

        pairs = [(query, doc) for doc in documents]
        scores = self._model.predict(pairs)

        results = [
            RerankResult(index=i, text=doc, score=float(score))
            for i, (doc, score) in enumerate(zip(documents, scores, strict=True))
        ]
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
