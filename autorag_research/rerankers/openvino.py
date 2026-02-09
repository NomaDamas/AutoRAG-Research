"""OpenVINO reranker implementation for Intel hardware-optimized inference."""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any

from pydantic import ConfigDict, Field

from autorag_research.rerankers.base import BaseReranker, RerankResult

logger = logging.getLogger("AutoRAG-Research")


class OpenVINOReranker(BaseReranker):
    """Reranker using OpenVINO for Intel hardware-optimized inference.

    Uses the optimum-intel library to run cross-encoder models with OpenVINO
    for optimized CPU inference on Intel hardware. Automatically exports
    models to OpenVINO format if not already available.

    Requires the `optimum-intel[openvino]` package:
    `pip install optimum-intel[openvino]`
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = Field(
        default="BAAI/bge-reranker-large",
        description="Model name from HuggingFace (auto-exported to OpenVINO format).",
    )
    max_length: int = Field(default=512, description="Maximum input sequence length.")

    _model: Any = None
    _tokenizer: Any = None

    def model_post_init(self, __context) -> None:
        """Initialize OpenVINO model and tokenizer after creation."""
        try:
            from optimum.intel.openvino import OVModelForSequenceClassification  # ty: ignore[unresolved-import]
            from transformers import AutoTokenizer
        except ImportError as e:
            msg = (
                "optimum-intel[openvino] and transformers packages are required. "
                "Install with: pip install optimum-intel[openvino] transformers"
            )
            raise ImportError(msg) from e

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = OVModelForSequenceClassification.from_pretrained(self.model_name, export=True)
        logger.info("Loaded OpenVINO reranker: %s", self.model_name)

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Apply sigmoid activation."""
        return 1.0 / (1.0 + math.exp(-x))

    def rerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        """Rerank documents using OpenVINO-optimized scoring.

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
        inputs = self._tokenizer(pairs, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")

        outputs = self._model(**inputs)
        raw_scores = outputs.logits.squeeze(-1).tolist()
        if isinstance(raw_scores, float):
            raw_scores = [raw_scores]

        results = [
            RerankResult(index=i, text=doc, score=self._sigmoid(score))
            for i, (doc, score) in enumerate(zip(documents, raw_scores, strict=True))
        ]
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    async def arerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        """Rerank documents asynchronously using OpenVINO-optimized scoring.

        Args:
            query: The search query.
            documents: List of document texts to rerank.
            top_k: Number of top results to return. If None, returns all documents.

        Returns:
            List of RerankResult objects sorted by relevance score (descending).
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.rerank, query, documents, top_k)
