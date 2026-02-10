"""Infinity Embedding API server client for late interaction (multi-vector) embeddings."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pydantic import Field, PrivateAttr

from autorag_research.embeddings.base import (
    MultiVectorEmbedding,
    MultiVectorMultiModalEmbedding,
)
from autorag_research.types import ImageType
from autorag_research.util import load_image

logger = logging.getLogger("AutoRAG-Research")


class InfinityEmbeddings(MultiVectorMultiModalEmbedding):
    """Client for Infinity Embedding API server running ColPali/ColQwen2 models.

    Uses the official ``infinity_client`` package (``InfinityVisionAPI``) which
    handles HTTP session management, retry with exponential backoff, base64
    decoding, numpy array reshaping, and semaphore-based concurrency control.

    Example:
        >>> embeddings = InfinityEmbeddings(
        ...     url="http://localhost:7997",
        ...     model_name="michaelfeil/colqwen2-v0.1",
        ... )
        >>> text_emb = embeddings.embed_text("Hello world")  # list[list[float]]
        >>> image_emb = embeddings.embed_image("image.png")  # list[list[float]]
    """

    url: str = Field(default="http://localhost:7997", description="Infinity API server URL.")
    model_name: str = Field(default="michaelfeil/colqwen2-v0.1", description="Model name served by Infinity.")
    encoding: str = Field(default="base64", description="Encoding format: 'base64' or 'float'.")

    _vision_client: Any = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Initialize the InfinityVisionAPI client after model creation."""
        from infinity_client.vision_client import InfinityVisionAPI

        self._vision_client = InfinityVisionAPI(url=self.url, format=self.encoding, model=self.model_name)  # ty: ignore[invalid-argument-type]

    def embed_text(self, text: str) -> MultiVectorEmbedding:
        """Embed a single text string.

        Args:
            text: The text to embed.

        Returns:
            Multi-vector embedding as list[list[float]].
        """
        embeddings, _tokens = self._vision_client.embed(self.model_name, [text]).result()
        return embeddings[0].tolist()

    async def aembed_text(self, text: str) -> MultiVectorEmbedding:
        """Embed a single text string asynchronously.

        Args:
            text: The text to embed.

        Returns:
            Multi-vector embedding as list[list[float]].
        """
        return await asyncio.to_thread(self.embed_text, text)

    def embed_query(self, query: str) -> MultiVectorEmbedding:
        """Embed a single query string.

        Args:
            query: The query to embed.

        Returns:
            Multi-vector embedding as list[list[float]].
        """
        return self.embed_text(query)

    async def aembed_query(self, query: str) -> MultiVectorEmbedding:
        """Embed a single query string asynchronously.

        Args:
            query: The query to embed.

        Returns:
            Multi-vector embedding as list[list[float]].
        """
        return await self.aembed_text(query)

    def embed_image(self, img_file_path: ImageType) -> MultiVectorEmbedding:
        """Embed a single image via the Infinity API.

        Args:
            img_file_path: Image as file path, bytes, or BytesIO.

        Returns:
            Multi-vector embedding as list[list[float]].
        """
        pil_img = load_image(img_file_path)
        embeddings, _tokens = self._vision_client.image_embed(self.model_name, [pil_img]).result()
        return embeddings[0].tolist()

    async def aembed_image(self, img_file_path: ImageType) -> MultiVectorEmbedding:
        """Embed a single image asynchronously via the Infinity API.

        Args:
            img_file_path: Image as file path, bytes, or BytesIO.

        Returns:
            Multi-vector embedding as list[list[float]].
        """
        return await asyncio.to_thread(self.embed_image, img_file_path)

    def embed_documents(self, texts: list[str]) -> list[MultiVectorEmbedding]:
        """Embed multiple documents in a single API call.

        Args:
            texts: List of texts to embed.

        Returns:
            List of multi-vector embeddings.
        """
        if not texts:
            return []
        embeddings, _tokens = self._vision_client.embed(self.model_name, texts).result()
        return [emb.tolist() for emb in embeddings]

    async def aembed_documents(self, texts: list[str]) -> list[MultiVectorEmbedding]:
        """Embed multiple documents asynchronously in a single API call.

        Args:
            texts: List of texts to embed.

        Returns:
            List of multi-vector embeddings.
        """
        if not texts:
            return []
        return await asyncio.to_thread(self.embed_documents, texts)

    def embed_images(self, img_file_paths: list[ImageType]) -> list[MultiVectorEmbedding]:
        """Embed multiple images in a single API call.

        Args:
            img_file_paths: List of image paths, bytes, or BytesIO objects.

        Returns:
            List of multi-vector embeddings.
        """
        if not img_file_paths:
            return []
        pil_images = [load_image(p) for p in img_file_paths]
        embeddings, _tokens = self._vision_client.image_embed(self.model_name, pil_images).result()
        return [emb.tolist() for emb in embeddings]

    async def aembed_images(self, img_file_paths: list[ImageType]) -> list[MultiVectorEmbedding]:
        """Embed multiple images asynchronously in a single API call.

        Args:
            img_file_paths: List of image paths, bytes, or BytesIO objects.

        Returns:
            List of multi-vector embeddings.
        """
        if not img_file_paths:
            return []
        return await asyncio.to_thread(self.embed_images, img_file_paths)
