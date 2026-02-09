"""KoReranker implementation for Korean-specific reranking."""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any

from pydantic import ConfigDict, Field

from autorag_research.rerankers.base import BaseReranker, RerankResult

logger = logging.getLogger("AutoRAG-Research")


class KoRerankerReranker(BaseReranker):
    """Reranker using KoReranker for Korean-specific document reranking.

    Uses a cross-encoder model fine-tuned for Korean text.
    Applies exponential normalization on logits to produce relevance scores.

    Requires `torch` and `transformers` packages.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = Field(
        default="Dongjin-kr/ko-reranker",
        description="KoReranker model name from HuggingFace.",
    )
    max_length: int = Field(default=512, description="Maximum input sequence length.")
    device: str | None = Field(default=None, description="Device to use (e.g. 'cuda', 'cpu'). Auto-detected if None.")

    _model: Any = None
    _tokenizer: Any = None
    _device: str = "cpu"

    def model_post_init(self, __context) -> None:
        """Initialize KoReranker model and tokenizer after creation."""
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:
            msg = "torch and transformers packages are required. Install with: pip install torch transformers"
            raise ImportError(msg) from e

        self._device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self._device)
        self._model.eval()
        logger.info("Loaded KoReranker: %s on %s", self.model_name, self._device)

    @staticmethod
    def _exp_normalize(scores: list[float]) -> list[float]:
        """Apply exponential normalization to scores."""
        max_score = max(scores) if scores else 0.0
        exp_scores = [math.exp(s - max_score) for s in scores]
        total = sum(exp_scores)
        if total == 0:
            return [0.0] * len(scores)
        return [s / total for s in exp_scores]

    def rerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        """Rerank documents using KoReranker scoring.

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

        import torch

        pairs = [(query, doc) for doc in documents]
        inputs = self._tokenizer(
            pairs, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        raw_scores = outputs.logits.squeeze(-1).tolist()
        if isinstance(raw_scores, float):
            raw_scores = [raw_scores]

        normalized_scores = self._exp_normalize(raw_scores)

        results = [
            RerankResult(index=i, text=doc, score=score)
            for i, (doc, score) in enumerate(zip(documents, normalized_scores, strict=True))
        ]
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    async def arerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        """Rerank documents asynchronously using KoReranker scoring.

        Args:
            query: The search query.
            documents: List of document texts to rerank.
            top_k: Number of top results to return. If None, returns all documents.

        Returns:
            List of RerankResult objects sorted by relevance score (descending).
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.rerank, query, documents, top_k)
