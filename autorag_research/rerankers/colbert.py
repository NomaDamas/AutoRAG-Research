"""ColBERT reranker implementation using token-level MaxSim scoring."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import Field

from autorag_research.rerankers.base import RerankResult
from autorag_research.rerankers.local_base import LocalReranker

logger = logging.getLogger("AutoRAG-Research")


class ColBERTReranker(LocalReranker):
    """Reranker using ColBERT token-level MaxSim scoring.

    ColBERT encodes queries and documents into token embeddings, then computes
    relevance as the average of maximum cosine similarities between query and
    document tokens.

    Requires `torch` and `transformers` packages.
    """

    model_name: str = Field(
        default="colbert-ir/colbertv2.0",
        description="ColBERT model name from HuggingFace.",
    )

    def model_post_init(self, __context) -> None:
        """Initialize ColBERT model and tokenizer after creation."""
        try:
            import torch  # noqa: F401
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            msg = "torch and transformers packages are required. Install with: pip install torch transformers"
            raise ImportError(msg) from e

        self._init_device()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self._device)
        self._model.eval()
        logger.info("Loaded ColBERT reranker: %s on %s", self.model_name, self._device)

    def _encode(self, texts: list[str]) -> Any:
        """Encode texts into token embeddings."""
        import torch

        inputs = self._tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Use last hidden state as token embeddings
        embeddings = outputs.last_hidden_state
        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        return embeddings, inputs["attention_mask"]

    def _maxsim_score(self, query_emb: Any, query_mask: Any, doc_emb: Any, doc_mask: Any) -> float:
        """Compute MaxSim score between query and document embeddings."""
        import torch

        # query_emb: [1, q_len, dim], doc_emb: [1, d_len, dim]
        # Compute similarity matrix: [q_len, d_len]
        sim_matrix = torch.matmul(query_emb.squeeze(0), doc_emb.squeeze(0).T)

        # Mask out padding tokens in document
        doc_mask_expanded = doc_mask.squeeze(0).unsqueeze(0).expand_as(sim_matrix)
        sim_matrix = sim_matrix.masked_fill(doc_mask_expanded == 0, float("-inf"))

        # Max over document tokens for each query token
        max_sim, _ = sim_matrix.max(dim=-1)  # [q_len]

        # Mask out padding tokens in query
        query_mask_squeezed = query_mask.squeeze(0).float()
        max_sim = max_sim * query_mask_squeezed

        # Average over valid query tokens
        score = max_sim.sum() / query_mask_squeezed.sum()
        return float(score.item())

    def rerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        """Rerank documents using ColBERT MaxSim scoring.

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

        # Encode query once
        query_emb, query_mask = self._encode([query])

        # Encode all documents in a single batch
        doc_embs, doc_masks = self._encode(documents)

        # Score each document using MaxSim
        results = []
        for i in range(len(documents)):
            score = self._maxsim_score(query_emb, query_mask, doc_embs[i : i + 1], doc_masks[i : i + 1])
            results.append(RerankResult(index=i, text=documents[i], score=score))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
