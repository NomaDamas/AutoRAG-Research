"""MonoT5 reranker implementation using sequence-to-sequence scoring."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pydantic import ConfigDict, Field

from autorag_research.rerankers.base import BaseReranker, RerankResult

logger = logging.getLogger("AutoRAG-Research")


class MonoT5Reranker(BaseReranker):
    """Reranker using MonoT5 sequence-to-sequence relevance scoring.

    MonoT5 uses a T5 model to classify query-document pairs as "true" (relevant)
    or "false" (not relevant). The probability of the "true" token is used as
    the relevance score.

    Requires `torch` and `transformers` packages.

    Reference: "Document Ranking with a Pretrained Sequence-to-Sequence Model"
    https://arxiv.org/abs/2003.06713
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = Field(
        default="castorini/monot5-3b-msmarco-10k",
        description="MonoT5 model name from HuggingFace.",
    )
    max_length: int = Field(default=512, description="Maximum input sequence length.")
    device: str | None = Field(default=None, description="Device to use (e.g. 'cuda', 'cpu'). Auto-detected if None.")

    _model: Any = None
    _tokenizer: Any = None
    _device: str = "cpu"
    _true_token_id: int = 0
    _false_token_id: int = 0

    def model_post_init(self, __context) -> None:
        """Initialize MonoT5 model and tokenizer after creation."""
        try:
            import torch
            from transformers import T5ForConditionalGeneration, T5Tokenizer
        except ImportError as e:
            msg = "torch and transformers packages are required. Install with: pip install torch transformers"
            raise ImportError(msg) from e

        self._device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self._model = T5ForConditionalGeneration.from_pretrained(self.model_name).to(self._device)
        self._model.eval()

        # Get token IDs for "true" and "false"
        self._true_token_id = self._tokenizer.encode("true", add_special_tokens=False)[0]
        self._false_token_id = self._tokenizer.encode("false", add_special_tokens=False)[0]
        logger.info("Loaded MonoT5 reranker: %s on %s", self.model_name, self._device)

    def _score_pair(self, query: str, document: str) -> float:
        """Score a single query-document pair."""
        import torch

        prompt = f"Query: {query} Document: {document} Relevant:"
        inputs = self._tokenizer(prompt, return_tensors="pt", max_length=self.max_length, truncation=True).to(
            self._device
        )

        with torch.no_grad():
            # Generate logits for the first output token
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=1,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # Get logits for the first generated token
        logits = outputs.scores[0][0]  # [vocab_size]

        # Softmax over true/false tokens
        true_false_logits = torch.tensor([logits[self._true_token_id], logits[self._false_token_id]])
        probs = torch.nn.functional.softmax(true_false_logits, dim=0)

        return float(probs[0].item())  # Probability of "true"

    def rerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        """Rerank documents using MonoT5 true/false scoring.

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

    async def arerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        """Rerank documents asynchronously using MonoT5 scoring.

        Args:
            query: The search query.
            documents: List of document texts to rerank.
            top_k: Number of top results to return. If None, returns all documents.

        Returns:
            List of RerankResult objects sorted by relevance score (descending).
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.rerank, query, documents, top_k)
