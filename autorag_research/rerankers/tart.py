"""TART (Task-Aware Retrieval with Instructions) reranker implementation."""

from __future__ import annotations

import logging

from pydantic import Field

from autorag_research.rerankers.base import RerankResult
from autorag_research.rerankers.local_base import LocalReranker

logger = logging.getLogger("AutoRAG-Research")


class TARTReranker(LocalReranker):
    """Reranker using TART (Task-Aware Retrieval with Instructions).

    TART uses instruction-following models for retrieval reranking.
    The query is prefixed with a task instruction, separated by [SEP],
    allowing the model to adapt its reranking behavior to different tasks.

    Requires `torch` and `transformers` packages.

    Reference: "Task-Aware Retrieval with Instructions"
    https://arxiv.org/abs/2211.09260
    """

    model_name: str = Field(
        default="facebook/tart-full-flan-t5-xl",
        description="TART model name from HuggingFace.",
    )
    instruction: str = Field(
        default="Find passage to answer given question",
        description="Task instruction for the TART model.",
    )

    def model_post_init(self, __context) -> None:
        """Initialize TART model and tokenizer after creation."""
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:
            msg = "torch and transformers packages are required. Install with: pip install torch transformers"
            raise ImportError(msg) from e

        self._init_device()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self._device)
        self._model.eval()
        logger.info("Loaded TART reranker: %s on %s", self.model_name, self._device)

    def _score_pair(self, query: str, document: str) -> float:
        """Score a single query-document pair with instruction."""
        import torch

        # Format: "{instruction} [SEP] {query}" paired with document
        instructed_query = f"{self.instruction} [SEP] {query}"

        inputs = self._tokenizer(
            instructed_query,
            document,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        logits = outputs.logits
        # Apply softmax and take the positive class probability
        probs = torch.nn.functional.softmax(logits, dim=-1)
        # Use the last class as the positive relevance score
        score = float(probs[0, -1].item())
        return score

    def rerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        """Rerank documents using TART instruction-aware scoring.

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

        results = []
        for i, doc in enumerate(documents):
            score = self._score_pair(query, doc)
            results.append(RerankResult(index=i, text=doc, score=score))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
