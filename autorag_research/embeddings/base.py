"""Base classes for multi-vector embeddings."""

import asyncio
from abc import abstractmethod

from llama_index.core.schema import ImageType
from pydantic import BaseModel, ConfigDict, Field

# Type alias for multi-vector embeddings (e.g., ColBERT, ColPali)
# Each text/image produces multiple vectors (one per token/patch)
MultiVectorEmbedding = list[list[float]]


class MultiVectorBaseEmbedding(BaseModel):
    """Base class for multi-vector text embeddings (e.g., ColBERT).

    Unlike single-vector embeddings, multi-vector models produce one vector
    per token, enabling late interaction for more fine-grained retrieval.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = Field(default="unknown", description="The embedding model name.")
    embed_batch_size: int = Field(default=10, description="Batch size for embedding.")

    @abstractmethod
    def get_query_embedding(self, query: str) -> MultiVectorEmbedding:
        """Get query embedding."""

    @abstractmethod
    async def aget_query_embedding(self, query: str) -> MultiVectorEmbedding:
        """Get query embedding asynchronously."""

    @abstractmethod
    def get_text_embedding(self, text: str) -> MultiVectorEmbedding:
        """Get text embedding."""

    @abstractmethod
    async def aget_text_embedding(self, text: str) -> MultiVectorEmbedding:
        """Get text embedding asynchronously."""

    def get_text_embeddings(self, texts: list[str]) -> list[MultiVectorEmbedding]:
        """Embed multiple texts. Subclasses can override for batch optimization."""
        return [self.get_text_embedding(text) for text in texts]

    async def aget_text_embeddings(self, texts: list[str]) -> list[MultiVectorEmbedding]:
        """Embed multiple texts asynchronously."""
        return await asyncio.gather(*[self.aget_text_embedding(t) for t in texts])

    def get_text_embedding_batch(self, texts: list[str], show_progress: bool = False) -> list[MultiVectorEmbedding]:
        """Get embeddings for multiple texts with batching."""
        results: list[MultiVectorEmbedding] = []
        for i in range(0, len(texts), self.embed_batch_size):
            batch = texts[i : i + self.embed_batch_size]
            results.extend(self.get_text_embeddings(batch))
        return results

    async def aget_text_embedding_batch(
        self, texts: list[str], show_progress: bool = False
    ) -> list[MultiVectorEmbedding]:
        """Get embeddings for multiple texts asynchronously with batching."""
        results: list[MultiVectorEmbedding] = []
        for i in range(0, len(texts), self.embed_batch_size):
            batch = texts[i : i + self.embed_batch_size]
            batch_results = await self.aget_text_embeddings(batch)
            results.extend(batch_results)
        return results


class MultiVectorMultiModalEmbedding(MultiVectorBaseEmbedding):
    """Base class for multi-vector multi-modal embeddings (e.g., ColPali, ColQwen2).

    Extends MultiVectorBaseEmbedding with image embedding support.
    Each image produces multiple vectors (one per patch/token).
    """

    @abstractmethod
    def get_image_embedding(self, img_file_path: ImageType) -> MultiVectorEmbedding:
        """Get image embedding."""

    @abstractmethod
    async def aget_image_embedding(self, img_file_path: ImageType) -> MultiVectorEmbedding:
        """Get image embedding asynchronously."""

    def get_image_embeddings(self, img_file_paths: list[ImageType]) -> list[MultiVectorEmbedding]:
        """Embed multiple images. Subclasses can override for batch optimization."""
        return [self.get_image_embedding(p) for p in img_file_paths]

    async def aget_image_embeddings(self, img_file_paths: list[ImageType]) -> list[MultiVectorEmbedding]:
        """Embed multiple images asynchronously."""
        return await asyncio.gather(*[self.aget_image_embedding(p) for p in img_file_paths])

    def get_image_embedding_batch(
        self, img_file_paths: list[ImageType], show_progress: bool = False
    ) -> list[MultiVectorEmbedding]:
        """Get embeddings for multiple images with batching."""
        results: list[MultiVectorEmbedding] = []
        for i in range(0, len(img_file_paths), self.embed_batch_size):
            batch = img_file_paths[i : i + self.embed_batch_size]
            results.extend(self.get_image_embeddings(batch))
        return results

    async def aget_image_embedding_batch(
        self, img_file_paths: list[ImageType], show_progress: bool = False
    ) -> list[MultiVectorEmbedding]:
        """Get embeddings for multiple images asynchronously with batching."""
        results: list[MultiVectorEmbedding] = []
        for i in range(0, len(img_file_paths), self.embed_batch_size):
            batch = img_file_paths[i : i + self.embed_batch_size]
            batch_results = await self.aget_image_embeddings(batch)
            results.extend(batch_results)
        return results
