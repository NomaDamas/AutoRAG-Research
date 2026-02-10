"""Base class for local model-based rerankers."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pydantic import Field

from autorag_research.rerankers.base import BaseReranker, RerankResult

logger = logging.getLogger("AutoRAG-Research")


class LocalReranker(BaseReranker):
    """Base class for all local model-based rerankers.

    All rerankers that run models locally (on CPU/GPU) should extend this class.
    Provides common fields (device, max_length) and async delegation via
    run_in_executor.

    For subclasses:
    - Call self._init_device() in model_post_init to auto-detect device
    - Implement rerank() with your scoring logic
    - arerank() is provided automatically via thread pool delegation
    """

    device: str | None = Field(default=None, description="Device to use (e.g. 'cuda', 'cpu'). Auto-detected if None.")
    max_length: int = Field(default=512, description="Maximum input sequence length.")

    _model: Any = None
    _tokenizer: Any = None
    _device: str = "cpu"

    def _init_device(self) -> str:
        """Auto-detect and set the compute device.

        Sets self._device based on self.device field or CUDA availability.

        Returns:
            The resolved device string.
        """
        try:
            import torch

            self._device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            self._device = self.device or "cpu"
        return self._device

    async def arerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        """Rerank documents asynchronously by delegating to sync rerank in executor.

        Args:
            query: The search query.
            documents: List of document texts to rerank.
            top_k: Number of top results to return. If None, returns all documents.

        Returns:
            List of RerankResult objects sorted by relevance score (descending).
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.rerank, query, documents, top_k)
