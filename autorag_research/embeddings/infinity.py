"""Infinity Embedding API server client for late interaction (multi-vector) embeddings."""

from __future__ import annotations

import asyncio
import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pydantic import Field, PrivateAttr
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from autorag_research.embeddings.base import (
    MultiVectorEmbedding,
    MultiVectorMultiModalEmbedding,
)
from autorag_research.types import ImageType

logger = logging.getLogger("AutoRAG-Research")


class InfinityEmbeddings(MultiVectorMultiModalEmbedding):
    """Client for Infinity Embedding API server running ColPali/ColQwen2 models.

    Connects to an Infinity server that exposes POST /embeddings with multi-vector
    (late interaction) embedding support. The server handles the GPU inference;
    this client only needs HTTP access.

    The API returns one embedding per input, where each embedding is a list of
    vectors (one per token/patch), suitable for MaxSim late interaction retrieval.

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
    hidden_dim: int = Field(default=128, description="Hidden dimension for reshaping flat arrays.")
    timeout: float = Field(default=60.0, description="HTTP request timeout in seconds.")
    max_retries: int = Field(default=3, description="Maximum number of retry attempts for API calls.")

    _client: Any = PrivateAttr(default=None)
    _async_client: Any = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Initialize HTTP clients after model creation."""
        import httpx

        self._client = httpx.Client(timeout=self.timeout)
        self._async_client = httpx.AsyncClient(timeout=self.timeout)

    def __del__(self) -> None:
        """Cleanup HTTP clients on deletion."""
        if self._client is not None:
            self._client.close()
        if self._async_client is not None:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._async_client.aclose())  # noqa: RUF006
                else:
                    loop.run_until_complete(self._async_client.aclose())
            except Exception:  # noqa: S110
                pass

    def _create_retry_decorator(self):
        """Create a retry decorator for API calls with exponential backoff."""
        return retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )

    @staticmethod
    def _image_to_base64_jpeg(img: ImageType) -> str:
        """Convert any ImageType to a base64-encoded JPEG string.

        Args:
            img: Image as file path (str/Path), raw bytes, or BytesIO.

        Returns:
            Base64-encoded JPEG string.
        """
        if isinstance(img, (str, Path)):
            image = Image.open(img).convert("RGB")
        elif isinstance(img, bytes):
            image = Image.open(BytesIO(img)).convert("RGB")
        elif isinstance(img, BytesIO):
            image = Image.open(img).convert("RGB")
        else:
            msg = f"Unsupported image type: {type(img)}. Expected str, Path, bytes, or BytesIO."
            raise TypeError(msg)

        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _parse_embeddings(self, response_data: dict) -> list[MultiVectorEmbedding]:
        """Parse API response into multi-vector embeddings.

        Args:
            response_data: JSON response from the Infinity API.

        Returns:
            List of multi-vector embeddings, each shaped (num_tokens, hidden_dim).
        """
        results: list[MultiVectorEmbedding] = []
        for item in response_data["data"]:
            embedding_data = item["embedding"]
            if self.encoding == "base64":
                raw_bytes = base64.b64decode(embedding_data)
                flat = np.frombuffer(raw_bytes, dtype=np.float32)
                multi_vec = flat.reshape(-1, self.hidden_dim).tolist()
            else:
                # Float encoding: already a flat list, reshape
                flat = np.array(embedding_data, dtype=np.float32)
                multi_vec = flat.reshape(-1, self.hidden_dim).tolist()
            results.append(multi_vec)
        return results

    def _call_api(self, inputs: list[str], modality: str) -> list[MultiVectorEmbedding]:
        """Make a synchronous API call to the Infinity server.

        Args:
            inputs: List of text strings or base64-encoded images.
            modality: Either "text" or "image".

        Returns:
            List of multi-vector embeddings.
        """

        @self._create_retry_decorator()
        def _request():
            response = self._client.post(
                f"{self.url}/embeddings",
                json={
                    "input": inputs,
                    "model": self.model_name,
                    "modality": modality,
                    "encoding": self.encoding,
                },
            )
            response.raise_for_status()
            return response.json()

        response_data = _request()
        return self._parse_embeddings(response_data)

    async def _acall_api(self, inputs: list[str], modality: str) -> list[MultiVectorEmbedding]:
        """Make an asynchronous API call to the Infinity server.

        Args:
            inputs: List of text strings or base64-encoded images.
            modality: Either "text" or "image".

        Returns:
            List of multi-vector embeddings.
        """

        @self._create_retry_decorator()
        async def _request():
            response = await self._async_client.post(
                f"{self.url}/embeddings",
                json={
                    "input": inputs,
                    "model": self.model_name,
                    "modality": modality,
                    "encoding": self.encoding,
                },
            )
            response.raise_for_status()
            return response.json()

        response_data = await _request()
        return self._parse_embeddings(response_data)

    def embed_text(self, text: str) -> MultiVectorEmbedding:
        """Embed a single text string.

        Args:
            text: The text to embed.

        Returns:
            Multi-vector embedding as list[list[float]].
        """
        results = self._call_api([text], modality="text")
        return results[0]

    async def aembed_text(self, text: str) -> MultiVectorEmbedding:
        """Embed a single text string asynchronously.

        Args:
            text: The text to embed.

        Returns:
            Multi-vector embedding as list[list[float]].
        """
        results = await self._acall_api([text], modality="text")
        return results[0]

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
        b64_image = self._image_to_base64_jpeg(img_file_path)
        results = self._call_api([b64_image], modality="image")
        return results[0]

    async def aembed_image(self, img_file_path: ImageType) -> MultiVectorEmbedding:
        """Embed a single image asynchronously via the Infinity API.

        Args:
            img_file_path: Image as file path, bytes, or BytesIO.

        Returns:
            Multi-vector embedding as list[list[float]].
        """
        b64_image = self._image_to_base64_jpeg(img_file_path)
        results = await self._acall_api([b64_image], modality="image")
        return results[0]

    def embed_documents(self, texts: list[str]) -> list[MultiVectorEmbedding]:
        """Embed multiple documents in a single API call.

        Args:
            texts: List of texts to embed.

        Returns:
            List of multi-vector embeddings.
        """
        if not texts:
            return []
        return self._call_api(texts, modality="text")

    async def aembed_documents(self, texts: list[str]) -> list[MultiVectorEmbedding]:
        """Embed multiple documents asynchronously in a single API call.

        Args:
            texts: List of texts to embed.

        Returns:
            List of multi-vector embeddings.
        """
        if not texts:
            return []
        return await self._acall_api(texts, modality="text")

    def embed_images(self, img_file_paths: list[ImageType]) -> list[MultiVectorEmbedding]:
        """Embed multiple images in a single API call.

        Args:
            img_file_paths: List of image paths, bytes, or BytesIO objects.

        Returns:
            List of multi-vector embeddings.
        """
        if not img_file_paths:
            return []
        b64_images = [self._image_to_base64_jpeg(p) for p in img_file_paths]
        return self._call_api(b64_images, modality="image")

    async def aembed_images(self, img_file_paths: list[ImageType]) -> list[MultiVectorEmbedding]:
        """Embed multiple images asynchronously in a single API call.

        Args:
            img_file_paths: List of image paths, bytes, or BytesIO objects.

        Returns:
            List of multi-vector embeddings.
        """
        if not img_file_paths:
            return []
        b64_images = [self._image_to_base64_jpeg(p) for p in img_file_paths]
        return await self._acall_api(b64_images, modality="image")
