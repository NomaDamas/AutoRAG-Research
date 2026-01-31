"""Base classes for multi-vector embeddings."""

import asyncio
from abc import abstractmethod

from pydantic import BaseModel, ConfigDict, Field

from autorag_research.types import ImageType

# Type alias for multi-vector embeddings (e.g., ColBERT, ColPali)
# Each text/image produces multiple vectors (one per token/patch)
MultiVectorEmbedding = list[list[float]]


class MultiVectorBaseEmbedding(BaseModel):
    """Base class for multi-vector text embeddings (e.g., ColBERT).

    Unlike single-vector embeddings, multi-vector models produce one vector
    per token, enabling late interaction for more fine-grained retrieval.

    Uses LangChain-style method names for consistency:
    - embed_query: Embed a single query
    - embed_documents: Embed multiple documents
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = Field(default="unknown", description="The embedding model name.")
    embed_batch_size: int = Field(default=10, description="Batch size for embedding.")

    @abstractmethod
    def embed_query(self, query: str) -> MultiVectorEmbedding:
        """Embed a single query."""

    @abstractmethod
    async def aembed_query(self, query: str) -> MultiVectorEmbedding:
        """Embed a single query asynchronously."""

    @abstractmethod
    def embed_text(self, text: str) -> MultiVectorEmbedding:
        """Embed a single text/document."""

    @abstractmethod
    async def aembed_text(self, text: str) -> MultiVectorEmbedding:
        """Embed a single text/document asynchronously."""

    def embed_documents(self, texts: list[str]) -> list[MultiVectorEmbedding]:
        """Embed multiple documents. Subclasses can override for batch optimization."""
        return [self.embed_text(text) for text in texts]

    async def aembed_documents(self, texts: list[str]) -> list[MultiVectorEmbedding]:
        """Embed multiple documents asynchronously."""
        return await asyncio.gather(*[self.aembed_text(t) for t in texts])

    def embed_documents_batch(self, texts: list[str], show_progress: bool = False) -> list[MultiVectorEmbedding]:
        """Get embeddings for multiple texts with batching."""
        results: list[MultiVectorEmbedding] = []
        for i in range(0, len(texts), self.embed_batch_size):
            batch = texts[i : i + self.embed_batch_size]
            results.extend(self.embed_documents(batch))
        return results

    async def aembed_documents_batch(self, texts: list[str], show_progress: bool = False) -> list[MultiVectorEmbedding]:
        """Get embeddings for multiple texts asynchronously with batching."""
        results: list[MultiVectorEmbedding] = []
        for i in range(0, len(texts), self.embed_batch_size):
            batch = texts[i : i + self.embed_batch_size]
            batch_results = await self.aembed_documents(batch)
            results.extend(batch_results)
        return results


class MultiVectorMultiModalEmbedding(MultiVectorBaseEmbedding):
    """Base class for multi-vector multi-modal embeddings (e.g., ColPali, ColQwen2).

    Extends MultiVectorBaseEmbedding with image embedding support.
    Each image produces multiple vectors (one per patch/token).
    """

    @abstractmethod
    def embed_image(self, img_file_path: ImageType) -> MultiVectorEmbedding:
        """Embed a single image."""

    @abstractmethod
    async def aembed_image(self, img_file_path: ImageType) -> MultiVectorEmbedding:
        """Embed a single image asynchronously."""

    def embed_images(self, img_file_paths: list[ImageType]) -> list[MultiVectorEmbedding]:
        """Embed multiple images. Subclasses can override for batch optimization."""
        return [self.embed_image(p) for p in img_file_paths]

    async def aembed_images(self, img_file_paths: list[ImageType]) -> list[MultiVectorEmbedding]:
        """Embed multiple images asynchronously."""
        return await asyncio.gather(*[self.aembed_image(p) for p in img_file_paths])

    def embed_images_batch(
        self, img_file_paths: list[ImageType], show_progress: bool = False
    ) -> list[MultiVectorEmbedding]:
        """Get embeddings for multiple images with batching."""
        results: list[MultiVectorEmbedding] = []
        for i in range(0, len(img_file_paths), self.embed_batch_size):
            batch = img_file_paths[i : i + self.embed_batch_size]
            results.extend(self.embed_images(batch))
        return results

    async def aembed_images_batch(
        self, img_file_paths: list[ImageType], show_progress: bool = False
    ) -> list[MultiVectorEmbedding]:
        """Get embeddings for multiple images asynchronously with batching."""
        results: list[MultiVectorEmbedding] = []
        for i in range(0, len(img_file_paths), self.embed_batch_size):
            batch = img_file_paths[i : i + self.embed_batch_size]
            batch_results = await self.aembed_images(batch)
            results.extend(batch_results)
        return results
