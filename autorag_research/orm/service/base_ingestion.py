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

    @abstractmethod
    def _get_schema_classes(self) -> dict[str, Any]:
        """Get schema classes from the schema namespace.

        Returns:
            Dictionary mapping class names to ORM classes.
        """
        ...

    # ==================== Embedding Operations ====================

    def _set_embeddings(
        self,
        entity_ids: list[int],
        embeddings: list,
        repo_attr: str,
        is_multi_vector: bool,
    ) -> int:
        """Generic helper to set embeddings for entities.

        Args:
            entity_ids: List of entity IDs.
            embeddings: List of embeddings (single or multi-vector).
            repo_attr: Repository attribute name (e.g., "queries", "chunks", "image_chunks").
            is_multi_vector: Whether to set embeddings (multi) or embedding (single).

        Returns:
            Number of entities successfully updated.

        Raises:
            LengthMismatchError: If entity_ids and embeddings have different lengths.
        """
        if len(entity_ids) != len(embeddings):
            raise LengthMismatchError("entity_ids", "embeddings")

        total_updated = 0

        with self._create_uow() as uow:
            if uow.session is None:
                raise SessionNotSetError

            repository = getattr(uow, repo_attr)
            for entity_id, embedding in zip(entity_ids, embeddings, strict=True):
                entity = repository.get_by_id(entity_id)
                if entity:
                    if is_multi_vector:
                        entity.embeddings = embedding
                    else:
                        entity.embedding = embedding
                    total_updated += 1
            uow.commit()

        return total_updated

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
        return self._set_embeddings(query_ids, embeddings, "queries", is_multi_vector=False)

    def set_query_multi_embeddings(
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
        return self._set_embeddings(query_ids, embeddings, "queries", is_multi_vector=True)

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
        return self._set_embeddings(chunk_ids, embeddings, "chunks", is_multi_vector=False)

    def set_chunk_multi_embeddings(
        self,
        chunk_ids: list[int],
        embeddings: list[list[list[float]]],
    ) -> int:
        """Batch set multi-vector embeddings for chunks.

        Args:
            chunk_ids: List of chunk IDs.
            embeddings: List of multi-vector embeddings (list of list of floats per chunk).
        Returns:
            Total number of chunks successfully updated.

        Raises:
            LengthMismatchError: If chunk_ids and embeddings have different lengths.
        """
        return self._set_embeddings(chunk_ids, embeddings, "chunks", is_multi_vector=True)

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

            # Filter out None embeddings and update entities
            valid_updates = [
                (item_id, emb) for (item_id, _), emb in zip(valid_items, embeddings, strict=True) if emb is not None
            ]
            if valid_updates:
                ids_to_update = [item_id for item_id, _ in valid_updates]
                embeddings_to_update = [emb for _, emb in valid_updates]
                total_embedded += self._set_embeddings(ids_to_update, embeddings_to_update, repo_attr, is_multi_vector)

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

        # ==================== Retrieval Ground Truth Operations ====================

    def add_retrieval_gt_batch(self, relations: list[dict]) -> list[tuple[int, int, int]]:
        """Batch add retrieval ground truth relations with explicit indices.

        Each relation must have exactly one of chunk_id or image_chunk_id.

        Args:
            relations: List of dicts with keys:
                      - query_id (int) - required
                      - chunk_id (int | None) - FK to Chunk (mutually exclusive with image_chunk_id)
                      - image_chunk_id (int | None) - FK to ImageChunk (mutually exclusive with chunk_id)
                      - group_index (int) - required
                      - group_order (int) - required

        Returns:
            List of created RetrievalRelation PKs as (query_id, group_index, group_order) tuples.

        Raises:
            ValueError: If both or neither of chunk_id/image_chunk_id are provided.
        """
        classes = self._get_schema_classes()
        RetrievalRelation = classes["RetrievalRelation"]

        # Validate mutual exclusivity
        for rel in relations:
            chunk_id = rel.get("chunk_id")
            image_chunk_id = rel.get("image_chunk_id")
            if (chunk_id is None) == (image_chunk_id is None):
                raise ValueError("Exactly one of chunk_id or image_chunk_id must be provided for each relation")  # noqa: TRY003

        with self._create_uow() as uow:
            if uow.session is None:
                raise SessionNotSetError
            relation_entities = [
                RetrievalRelation(
                    query_id=rel["query_id"],
                    chunk_id=rel.get("chunk_id"),
                    image_chunk_id=rel.get("image_chunk_id"),
                    group_index=rel["group_index"],
                    group_order=rel["group_order"],
                )
                for rel in relations
            ]
            uow.retrieval_relations.add_all(relation_entities)
            uow.flush()
            pks = [(r.query_id, r.group_index, r.group_order) for r in relation_entities]
            uow.commit()
            return pks

    def add_retrieval_gt_multihop(
        self,
        query_id: int,
        groups: list[list[tuple[str, int]]],
    ) -> list[tuple[int, int, int]]:
        """Add mixed multi-hop retrieval ground truth (text and image chunks in same chain).

        Each group represents a "hop" in the retrieval chain. Items within each group
        can be either text chunks or image chunks.

        Args:
            query_id: The query ID.
            groups: List of groups, where each group is a list of (type, id) tuples.
                   type: "chunk" for text chunks, "image_chunk" for image chunks.
                   id: The chunk ID or image chunk ID.

                   Example:
                   [
                       [("chunk", 1), ("chunk", 2)],       # First hop: text chunks
                       [("image_chunk", 1)],               # Second hop: image chunk
                       [("chunk", 3), ("image_chunk", 2)], # Third hop: mixed
                   ]

        Returns:
            List of created RetrievalRelation PKs as (query_id, group_index, group_order) tuples.

        Raises:
            ValueError: If type is not "chunk" or "image_chunk".
        """
        classes = self._get_schema_classes()
        RetrievalRelation = classes["RetrievalRelation"]

        with self._create_uow() as uow:
            if uow.session is None:
                raise SessionNotSetError

            # Get current max group index for this query
            max_group_idx = uow.retrieval_relations.get_max_group_index(query_id)
            start_group_idx = (max_group_idx or -1) + 1

            all_relations = []
            for group_offset, group_items in enumerate(groups):
                group_index = start_group_idx + group_offset
                for order, (item_type, item_id) in enumerate(group_items):
                    if item_type == "chunk":
                        relation = RetrievalRelation(
                            query_id=query_id,
                            chunk_id=item_id,
                            image_chunk_id=None,
                            group_index=group_index,
                            group_order=order,
                        )
                    elif item_type == "image_chunk":
                        relation = RetrievalRelation(
                            query_id=query_id,
                            chunk_id=None,
                            image_chunk_id=item_id,
                            group_index=group_index,
                            group_order=order,
                        )
                    else:
                        raise ValueError(f"Invalid item type: {item_type}. Must be 'chunk' or 'image_chunk'.")  # noqa: TRY003
                    all_relations.append(relation)

            uow.retrieval_relations.add_all(all_relations)
            uow.flush()
            pks = [(r.query_id, r.group_index, r.group_order) for r in all_relations]
            uow.commit()
            return pks
