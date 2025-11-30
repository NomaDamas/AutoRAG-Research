"""Base Ingestion Service for AutoRAG-Research.

Provides common functionality for data ingestion services including
query/chunk embedding operations with async batch processing.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from autorag_research.exceptions import LengthMismatchError, SessionNotSetError

EmbeddingFunc = Callable[[str], Awaitable[list[float]]]

logger = logging.getLogger("AutoRAG-Research")


class BaseIngestionService(ABC):
    """Abstract base class for data ingestion services.

    Provides common functionality for:
    - Setting embeddings for queries and chunks
    - Async batch embedding with concurrency control

    Subclasses must implement:
    - _create_uow() to return their specific UoW type
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        schema: Any | None = None,
    ):
        """Initialize the ingestion service.

        Args:
            session_factory: SQLAlchemy sessionmaker for database connections.
            schema: Schema namespace from create_schema(). If None, uses default 768-dim schema.
        """
        self.session_factory = session_factory
        self._schema = schema

    @abstractmethod
    def _create_uow(self) -> Any:
        """Create a new Unit of Work instance.

        Subclasses must implement this to return their specific UoW type.
        """
        ...

    # ==================== Embedding Operations ====================

    def set_query_embeddings(
        self,
        query_ids: list[int],
        embeddings: list[list[float]],
    ) -> int:
        """Set embeddings for multiple queries.

        Args:
            query_ids: List of query IDs to set embeddings for.
            embeddings: List of embedding vectors (must match query_ids length).

        Returns:
            Total number of queries successfully updated.

        Raises:
            LengthMismatchError: If query_ids and embeddings have different lengths.
        """
        if len(query_ids) != len(embeddings):
            raise LengthMismatchError("query_ids", "embeddings")

        total_updated = 0

        with self._create_uow() as uow:
            if uow.session is None:
                raise SessionNotSetError
            for query_id, embedding in zip(query_ids, embeddings, strict=True):
                query = uow.queries.get_by_id(query_id)
                if query:
                    query.embedding = embedding
                    total_updated += 1
            uow.commit()

        return total_updated

    def set_chunk_embeddings(
        self,
        chunk_ids: list[int],
        embeddings: list[list[float]],
    ) -> int:
        """Set embeddings for multiple chunks.

        Args:
            chunk_ids: List of chunk IDs to set embeddings for.
            embeddings: List of embedding vectors (must match chunk_ids length).

        Returns:
            Total number of chunks successfully updated.

        Raises:
            LengthMismatchError: If chunk_ids and embeddings have different lengths.
        """
        if len(chunk_ids) != len(embeddings):
            raise LengthMismatchError("chunk_ids", "embeddings")

        total_updated = 0

        with self._create_uow() as uow:
            if uow.session is None:
                raise SessionNotSetError
            for chunk_id, embedding in zip(chunk_ids, embeddings, strict=True):
                chunk = uow.chunks.get_by_id(chunk_id)
                if chunk:
                    chunk.embedding = embedding
                    total_updated += 1
            uow.commit()

        return total_updated

    # ==================== Batch Embedding Operations ====================

    def embed_all_queries(
        self,
        embed_func: EmbeddingFunc,
        batch_size: int = 100,
        max_concurrency: int = 10,
    ) -> int:
        """Embed all queries that don't have embeddings.

        Processes queries in batches with concurrent embedding calls.

        Args:
            embed_func: Async function that takes query text and returns embedding vector.
            batch_size: Number of queries to process per batch.
            max_concurrency: Maximum concurrent embedding calls.

        Returns:
            Total number of queries successfully embedded.
        """
        total_embedded = 0

        while True:
            with self._create_uow() as uow:
                queries = uow.queries.get_queries_without_embeddings(limit=batch_size)
                if not queries:
                    break
                items_to_embed = [(q.id, q.query) for q in queries]

            embeddings = asyncio.run(self._embed_text_batch(items_to_embed, embed_func, max_concurrency))

            with self._create_uow() as uow:
                if uow.session is None:
                    raise SessionNotSetError
                for (item_id, _), embedding in zip(items_to_embed, embeddings, strict=True):
                    query = uow.queries.get_by_id(item_id)
                    if query and embedding is not None:
                        query.embedding = embedding
                        total_embedded += 1
                uow.commit()

        return total_embedded

    def embed_all_chunks(
        self,
        embed_func: EmbeddingFunc,
        batch_size: int = 100,
        max_concurrency: int = 10,
    ) -> int:
        """Embed all chunks that don't have embeddings.

        Processes chunks in batches with concurrent embedding calls.

        Args:
            embed_func: Async function that takes chunk text and returns embedding vector.
            batch_size: Number of chunks to process per batch.
            max_concurrency: Maximum concurrent embedding calls.

        Returns:
            Total number of chunks successfully embedded.
        """
        total_embedded = 0

        while True:
            with self._create_uow() as uow:
                chunks = uow.chunks.get_chunks_without_embeddings(limit=batch_size)
                if not chunks:
                    break
                items_to_embed = [(c.id, c.contents) for c in chunks]

            embeddings = asyncio.run(self._embed_text_batch(items_to_embed, embed_func, max_concurrency))

            with self._create_uow() as uow:
                if uow.session is None:
                    raise SessionNotSetError
                for (item_id, _), embedding in zip(items_to_embed, embeddings, strict=True):
                    chunk = uow.chunks.get_by_id(item_id)
                    if chunk and embedding is not None:
                        chunk.embedding = embedding
                        total_embedded += 1
                uow.commit()

        return total_embedded

    @staticmethod
    async def _embed_text_batch(
        items: list[tuple[int, str]],
        embed_func: EmbeddingFunc,
        max_concurrency: int,
    ) -> list[list[float] | None]:
        """Embed a batch of text items with concurrency control.

        Args:
            items: List of (id, text) tuples.
            embed_func: Async function that takes text and returns embedding.
            max_concurrency: Maximum concurrent calls.

        Returns:
            List of embeddings (or None if failed) in same order as items.
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def embed_with_semaphore(text: str) -> list[float] | None:
            async with semaphore:
                try:
                    return await embed_func(text)
                except Exception:
                    logger.exception(f"Failed to embed text: {text[:50]}...")
                    return None

        tasks = [embed_with_semaphore(text) for _, text in items]
        return await asyncio.gather(*tasks)
