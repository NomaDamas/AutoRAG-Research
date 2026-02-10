"""MonoT5 reranker implementation using sequence-to-sequence scoring."""

from __future__ import annotations

import logging

from pydantic import Field

from autorag_research.rerankers.base import RerankResult
from autorag_research.rerankers.local_base import LocalReranker

logger = logging.getLogger("AutoRAG-Research")


class MonoT5Reranker(LocalReranker):
    """Reranker using MonoT5 sequence-to-sequence relevance scoring.

    MonoT5 uses a T5 model to classify query-document pairs as "true" (relevant)
    or "false" (not relevant). The probability of the "true" token is used as
    the relevance score.

    Requires `torch` and `transformers` packages.

    Reference: "Document Ranking with a Pretrained Sequence-to-Sequence Model"
    https://arxiv.org/abs/2003.06713
    """

    model_name: str = Field(
        default="castorini/monot5-3b-msmarco-10k",
        description="MonoT5 model name from HuggingFace.",
    )

    _true_token_id: int = 0
    _false_token_id: int = 0

    def model_post_init(self, __context) -> None:
        """Initialize MonoT5 model and tokenizer after creation."""
        try:
            import torch  # noqa: F401
            from transformers import T5ForConditionalGeneration, T5Tokenizer
        except ImportError as e:
            msg = "torch and transformers packages are required. Install with: pip install torch transformers"
            raise ImportError(msg) from e

        self._init_device()
        self._tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self._model = T5ForConditionalGeneration.from_pretrained(self.model_name).to(self._device)
        self._model.eval()

        # Get token IDs for "true" and "false"
        self._true_token_id = self._tokenizer.encode("true", add_special_tokens=False)[0]
        self._false_token_id = self._tokenizer.encode("false", add_special_tokens=False)[0]
        logger.info("Loaded MonoT5 reranker: %s on %s", self.model_name, self._device)

    def rerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        """Rerank documents using MonoT5 batch true/false scoring.

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

        prompts = [f"Query: {query} Document: {doc} Relevant:" for doc in documents]

        scores: list[float] = []
        for batch_start in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[batch_start : batch_start + self.batch_size]
            inputs = self._tokenizer(
                batch_prompts, return_tensors="pt", max_length=self.max_length, truncation=True, padding=True
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            # outputs.scores[0] shape: [batch_size, vocab_size]
            logits = outputs.scores[0]
            true_false = logits[:, [self._true_token_id, self._false_token_id]]
            probs = torch.nn.functional.softmax(true_false, dim=-1)
            scores.extend(probs[:, 0].tolist())

        results = [
            RerankResult(index=i, text=doc, score=score)
            for i, (doc, score) in enumerate(zip(documents, scores, strict=True))
        ]
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
