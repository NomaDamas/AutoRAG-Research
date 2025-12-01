"""Base Ingestion Service for AutoRAG-Research.

Provides common functionality for data ingestion services including
query/chunk embedding operations with async batch processing.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from sqlalchemy.orm import Session, sessionmaker

from autorag_research.exceptions import LengthMismatchError, RepositoryNotSupportedError, SessionNotSetError
from autorag_research.util import run_with_concurrency_limit

# Type alias for embedding functions
# Single-vector embedding functions
ImageEmbeddingFunc = Callable[[bytes], Awaitable[list[float]]]
TextEmbeddingFunc = Callable[[str], Awaitable[list[float]]]
# Multi-vector embedding functions (for late interaction models like ColPali)
ImageMultiVectorEmbeddingFunc = Callable[[bytes], Awaitable[list[list[float]]]]
TextMultiVectorEmbeddingFunc = Callable[[str], Awaitable[list[list[float]]]]

# Literal types for entity and embedding types
EntityType = Literal["query", "chunk", "image_chunk"]
EmbeddingType = Literal["single", "multi_vector"]

logger = logging.getLogger("AutoRAG-Research")

# Entity configuration for _embed_entities method
# Maps entity_type to (repository_attr, data_attr, display_name, filter_none)
ENTITY_CONFIG: dict[str, tuple[str, str, str, bool]] = {
    "query": ("queries", "query", "queries", False),
    "chunk": ("chunks", "contents", "chunks", False),
    "image_chunk": ("image_chunks", "content", "image chunks", True),
}


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

    def set_query_multi_vector_embeddings(
        self,
        query_ids: list[int],
        embeddings: list[list[list[float]]],
    ) -> int:
        """Batch set multi-vector embeddings for queries.

        Args:
            query_ids: List of query IDs.
            embeddings: List of multi-vector embeddings (list of list of floats per query).

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
                    query.embeddings = embedding
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

    def _embed_entities(
        self,
        entity_type: EntityType,
        embedding_type: EmbeddingType,
        embed_func: ImageEmbeddingFunc
        | TextEmbeddingFunc
        | ImageMultiVectorEmbeddingFunc
        | TextMultiVectorEmbeddingFunc,
        batch_size: int,
        max_concurrency: int,
    ) -> int:
        """Generic method to embed entities (queries, chunks, or image chunks) with single or multi-vector embeddings.

        Args:
            entity_type: Type of entity to embed ("query", "chunk", or "image_chunk").
            embedding_type: Type of embedding ("single" or "multi_vector").
            embed_func: Async function that takes data and returns embedding.
            batch_size: Number of entities to process per batch.
            max_concurrency: Maximum concurrent embedding calls.

        Returns:
            Total number of entities successfully embedded.

        Raises:
            RepositoryNotSupportedError: If the required repository is not available in the UoW.
        """
        # Get entity configuration from dictionary
        repo_attr, data_attr, display_name, filter_none = ENTITY_CONFIG[entity_type]
        fetch_method_name = "get_without_embeddings" if embedding_type == "single" else "get_without_multi_embeddings"
        is_multi_vector = embedding_type == "multi_vector"

        # Determine log/error messages
        embedding_suffix = " with multi-vector" if is_multi_vector else ""
        is_image = entity_type == "image_chunk"
        error_msg = f"Failed to embed {'image' if is_image else 'text'}{embedding_suffix}"

        total_embedded = 0

        while True:
            # Fetch entities without embeddings
            with self._create_uow() as uow:
                # Check if repository exists
                repository = getattr(uow, repo_attr, None)
                if repository is None:
                    raise RepositoryNotSupportedError(repo_attr, type(uow).__name__)

                # Get fetch method and call it
                fetch_method = getattr(repository, fetch_method_name)
                entities = fetch_method(limit=batch_size)
                if not entities:
                    break

                # Extract (id, data) pairs using data_attr
                items_to_embed = [(e.id, getattr(e, data_attr)) for e in entities]

            # Filter None content if required (for image chunks)
            if filter_none:
                valid_items = [(item_id, data) for item_id, data in items_to_embed if data is not None]
                if not valid_items:
                    break
            else:
                valid_items = items_to_embed

            # Run embedding
            data_list = [data for _, data in valid_items]
            embeddings = asyncio.run(run_with_concurrency_limit(data_list, embed_func, max_concurrency, error_msg))

            # Update entities with embeddings
            with self._create_uow() as uow:
                if uow.session is None:
                    raise SessionNotSetError

                repository = getattr(uow, repo_attr)
                for (item_id, _), embedding in zip(valid_items, embeddings, strict=True):
                    entity = repository.get_by_id(item_id)
                    if entity and embedding is not None:
                        if is_multi_vector:
                            entity.embeddings = embedding
                        else:
                            entity.embedding = embedding
                        total_embedded += 1
                uow.commit()

            logger.info(f"Embedded {total_embedded} {display_name}{embedding_suffix} so far")

        logger.info(f"Total {display_name} embedded{embedding_suffix}: {total_embedded}")
        return total_embedded

    def embed_all_queries(
        self,
        embed_func: TextEmbeddingFunc,
        batch_size: int = 100,
        max_concurrency: int = 10,
    ) -> int:
        """Embed all queries that don't have embeddings.

        Args:
            embed_func: Async function that takes query text and returns embedding vector.
            batch_size: Number of queries to process per batch.
            max_concurrency: Maximum concurrent embedding calls.

        Returns:
            Total number of queries successfully embedded.
        """
        return self._embed_entities("query", "single", embed_func, batch_size, max_concurrency)

    def embed_all_chunks(
        self,
        embed_func: TextEmbeddingFunc,
        batch_size: int = 100,
        max_concurrency: int = 10,
    ) -> int:
        """Embed all chunks that don't have embeddings.

        Args:
            embed_func: Async function that takes chunk text and returns embedding vector.
            batch_size: Number of chunks to process per batch.
            max_concurrency: Maximum concurrent embedding calls.

        Returns:
            Total number of chunks successfully embedded.
        """
        return self._embed_entities("chunk", "single", embed_func, batch_size, max_concurrency)
