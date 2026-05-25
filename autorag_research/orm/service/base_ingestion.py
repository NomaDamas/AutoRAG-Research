"""Base Ingestion Service for AutoRAG-Research.

Provides common functionality for data ingestion services including
query/chunk embedding operations with async batch processing.
"""

import asyncio
import logging
from abc import ABC
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

from autorag_research.exceptions import (
    DuplicateRetrievalGTError,
    LengthMismatchError,
    RepositoryNotSupportedError,
    SessionNotSetError,
)
from autorag_research.orm.service.base import BaseService
from autorag_research.util import run_with_concurrency_limit

# Type alias for embedding functions
# Single-vector embedding functions (supports LangChain ImageType: str | bytes | Path)
ImageEmbeddingFunc = Callable[[str | bytes | Path | BytesIO], Awaitable[list[float]]]
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
    "query": ("queries", "contents", "queries", False),
    "chunk": ("chunks", "contents", "chunks", False),
    "image_chunk": ("image_chunks", "content", "image chunks", True),
}


def _partition_none_content(
    items: list[tuple[int | str, Any]],
) -> tuple[list[tuple[int | str, Any]], list[tuple[int | str, Any]]]:
    """Split items into (with-content, none-content) lists for image_chunk filter."""
    valid: list[tuple[int | str, Any]] = []
    none_content: list[tuple[int | str, Any]] = []
    for item_id, data in items:
        (none_content if data is None else valid).append((item_id, data))
    return valid, none_content


def _partition_embedding_results(
    items: list[tuple[int | str, Any]],
    embeddings: list[Any],
) -> tuple[list[tuple[int | str, Any]], list[int | str]]:
    """Split (item, embedding) pairs into (successful_updates, failed_ids)."""
    valid_updates: list[tuple[int | str, Any]] = []
    failed_ids: list[int | str] = []
    for (item_id, _), emb in zip(items, embeddings, strict=True):
        if emb is None:
            failed_ids.append(item_id)
        else:
            valid_updates.append((item_id, emb))
    return valid_updates, failed_ids


@dataclass
class _EmbedRunState:
    """Mutable counters tracked across one `_embed_entities` run."""

    total_embedded: int = 0
    skipped_none_content: int = 0


class BaseIngestionService(BaseService, ABC):
    """Abstract base class for data ingestion services.

    Provides common functionality for:
    - Setting embeddings for queries and chunks
    - Async batch embedding with concurrency control

    Subclasses must implement:
    - _create_uow() to return their specific UoW type
    """

    def add_chunks(
        self, chunks: list[dict[str, str | int | bool | None]], skip_duplicates: bool = True
    ) -> list[int | str]:
        """Batch add text chunks to the database.

        Uses memory-efficient bulk insert (SQLAlchemy Core) instead of ORM objects.
        This reduces memory usage by ~3-5x for large batches.

        Args:
            chunks: List of dict with keys:
                - id (optional): Chunk ID
                - contents (required): Text content
                - is_table (optional, default=False): Whether this chunk is a table
                - table_type (optional): Table format type (markdown, xml, html)
            skip_duplicates: If True, silently skip chunks with duplicate primary keys
                instead of raising an IntegrityError.
                Default is True.

        Returns:
            List of created Chunk IDs (excludes skipped duplicates when skip_duplicates=True).
        """
        return self._add_bulk(chunks, repository_property="chunks", skip_duplicates=skip_duplicates)

    def link_pages_to_chunks(
        self,
        relations: list[dict[str, int | str]],
    ) -> list[tuple[int | str, int | str]]:
        """Batch create Page-Chunk relations (M:N relationship support).

        This is the primary method for creating Page-Chunk relations.
        Use this when you need to link multiple pages to multiple chunks in a single call.

        Args:
            relations: List of dict with keys:
                - page_id (required): Page ID
                - chunk_id (required): Chunk ID

        Returns:
            List of created PageChunkRelation PKs as (page_id, chunk_id) tuples.

        Raises:
            RepositoryNotSupportedError: If PageChunkRelation is not available in schema.
            SessionNotSetError: If UoW session is not initialized.
        """
        classes = self._get_schema_classes()
        PageChunkRelation = classes.get("PageChunkRelation")
        if PageChunkRelation is None:
            raise RepositoryNotSupportedError("PageChunkRelation", "schema")

        with self._create_uow() as uow:
            if uow.session is None:
                raise SessionNotSetError
            relation_entities = [
                PageChunkRelation(page_id=rel["page_id"], chunk_id=rel["chunk_id"]) for rel in relations
            ]
            for entity in relation_entities:
                uow.session.add(entity)
            uow.flush()
            pks = [(r.page_id, r.chunk_id) for r in relation_entities]
            uow.commit()
            return pks

    def link_page_to_chunks(
        self,
        page_id: int | str,
        chunk_ids: list[int] | list[str],
    ) -> list[tuple[int | str, int | str]]:
        """Link a single Page to multiple Chunks (1:N relationship).

        Convenience method for the common case where one page contains multiple chunks.
        Internally calls link_pages_to_chunks.

        Args:
            page_id: The Page ID to link from.
            chunk_ids: List of Chunk IDs to link to.

        Returns:
            List of created PageChunkRelation PKs as (page_id, chunk_id) tuples.

        Raises:
            RepositoryNotSupportedError: If PageChunkRelation is not available in schema.
            SessionNotSetError: If UoW session is not initialized.
        """
        return self.link_pages_to_chunks([{"page_id": page_id, "chunk_id": cid} for cid in chunk_ids])

    def add_queries(
        self, queries: list[dict[str, str | list[str] | None]], skip_duplicates: bool = True
    ) -> list[int | str]:
        """Batch add queries to the database.

        Uses memory-efficient bulk insert (SQLAlchemy Core) instead of ORM objects.
        This reduces memory usage by ~3-5x for large batches.

        Args:
            queries: List of dict with keys: id (optional), contents, generation_gt (optional).
            skip_duplicates: If True, silently skip queries with duplicate primary keys
                instead of raising an IntegrityError.
                Default is True.

        Returns:
            List of created Query IDs (excludes skipped duplicates when skip_duplicates=True).
        """
        return self._add_bulk(queries, repository_property="queries", skip_duplicates=skip_duplicates)

    # ==================== Embedding Operations ====================

    def _set_embeddings(
        self,
        entity_ids: list[int | str],
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

            if is_multi_vector:
                # Use repository's raw SQL method for multi-vector embeddings
                # to bypass pgvector's array type processing issue
                total_updated = repository.set_multi_vector_embeddings_batch(
                    entity_ids, embeddings, vector_column="embeddings", id_column="id"
                )
            else:
                # Single-vector embeddings work fine with ORM
                for entity_id, embedding in zip(entity_ids, embeddings, strict=True):
                    entity = repository.get_by_id(entity_id)
                    if entity:
                        entity.embedding = embedding
                        total_updated += 1

            uow.commit()

        return total_updated

    def set_query_embeddings(
        self,
        query_ids: list[int | str],
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
        query_ids: list[int | str],
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
        chunk_ids: list[int | str],
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
        chunk_ids: list[int | str],
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
        bm25_tokenizer: str | None = "bert",
    ) -> int:
        """Generic method to embed entities (queries, chunks, or image chunks) with single or multi-vector embeddings.

        Args:
            entity_type: Type of entity to embed ("query", "chunk", or "image_chunk").
            embedding_type: Type of embedding ("single" or "multi_vector").
            embed_func: Async function that takes data and returns embedding.
            batch_size: Number of entities to process per batch.
            max_concurrency: Maximum concurrent embedding calls.
            bm25_tokenizer: Tokenizer for BM25 tokens (only for chunks/queries). Set to None to skip.
                Available tokenizers (pg_tokenizer pre-built models):
                    - "bert": bert-base-uncased (Hugging Face) - Default
                    - "wiki_tocken": Wikitext-103 trained model
                    - "gemma2b": Google lightweight model (~100MB memory)
                    - "llmlingua2": Microsoft summarization model (~200MB memory)
                See: https://github.com/tensorchord/pg_tokenizer.rs/blob/main/docs/06-model.md

        Returns:
            Total number of entities successfully embedded.

        Raises:
            RepositoryNotSupportedError: If the required repository is not available in the UoW.
        """
        from tqdm import tqdm

        # Get entity configuration from dictionary
        repo_attr, data_attr, display_name, filter_none = ENTITY_CONFIG[entity_type]
        fetch_method_name = "get_without_embeddings" if embedding_type == "single" else "get_without_multi_embeddings"
        count_method_name = (
            "count_without_embeddings" if embedding_type == "single" else "count_without_multi_embeddings"
        )
        is_multi_vector = embedding_type == "multi_vector"

        # Determine log/error messages
        embedding_suffix = " with multi-vector" if is_multi_vector else ""
        is_image = entity_type == "image_chunk"
        error_msg = f"Failed to embed {'image' if is_image else 'text'}{embedding_suffix}"

        # Get total count for progress bar
        with self._create_uow() as uow:
            repository = getattr(uow, repo_attr, None)
            if repository is None:
                raise RepositoryNotSupportedError(repo_attr, type(uow).__name__)
            total_to_embed = getattr(repository, count_method_name)()

        if total_to_embed == 0:
            logger.info(f"No {display_name} to embed{embedding_suffix}")
            return 0

        # `failed_ids` is per-run only — junk rows that fail in this run are skipped
        # for the rest of the run to prevent re-fetching them forever (the same query
        # returns "rows without embeddings" indefinitely). A subsequent ingest call
        # will retry them, which is correct for transient failures.
        failed_ids: set[int | str] = set()
        state = _EmbedRunState()

        with tqdm(total=total_to_embed, desc=f"Embedding {display_name}", unit="items") as pbar:
            while True:
                items_to_embed = self._fetch_unembedded_batch(
                    repo_attr=repo_attr,
                    fetch_method_name=fetch_method_name,
                    data_attr=data_attr,
                    batch_size=batch_size,
                    failed_ids=failed_ids,
                )
                if not items_to_embed:
                    break

                if filter_none:
                    items_to_embed, none_skipped = _partition_none_content(items_to_embed)
                    if none_skipped:
                        failed_ids.update(item_id for item_id, _ in none_skipped)
                        state.skipped_none_content += len(none_skipped)
                    if not items_to_embed:
                        continue

                items_to_embed = items_to_embed[:batch_size]
                made_progress = self._embed_batch(
                    items_to_embed=items_to_embed,
                    embed_func=embed_func,
                    max_concurrency=max_concurrency,
                    error_msg=error_msg,
                    display_name=display_name,
                    repo_attr=repo_attr,
                    is_multi_vector=is_multi_vector,
                    failed_ids=failed_ids,
                    state=state,
                    pbar=pbar,
                )
                if not made_progress:
                    break

        if entity_type in ("chunk", "query") and bm25_tokenizer is not None:
            self._populate_bm25_tokens(bm25_tokenizer, entity_type=entity_type, batch_size=batch_size)

        embed_failures = len(failed_ids) - state.skipped_none_content
        logger.info(
            f"Total {display_name} embedded{embedding_suffix}: {state.total_embedded} "
            f"(skipped_failed={embed_failures}, skipped_empty_content={state.skipped_none_content})"
        )
        return state.total_embedded

    def _fetch_unembedded_batch(
        self,
        *,
        repo_attr: str,
        fetch_method_name: str,
        data_attr: str,
        batch_size: int,
        failed_ids: set[int | str],
    ) -> list[tuple[int | str, Any]]:
        """Fetch the next batch of entities without embeddings, excluding `failed_ids`.

        Over-fetches by `len(failed_ids)` so already-failed rows do not starve
        the batch when they appear at the head of the database ordering.
        """
        with self._create_uow() as uow:
            repository = getattr(uow, repo_attr, None)
            if repository is None:
                raise RepositoryNotSupportedError(repo_attr, type(uow).__name__)
            fetch_method = getattr(repository, fetch_method_name)
            entities = fetch_method(limit=batch_size + len(failed_ids))
            if not entities:
                return []
            return [(e.id, getattr(e, data_attr)) for e in entities if e.id not in failed_ids]

    def _embed_batch(
        self,
        *,
        items_to_embed: list[tuple[int | str, Any]],
        embed_func: Any,
        max_concurrency: int,
        error_msg: str,
        display_name: str,
        repo_attr: str,
        is_multi_vector: bool,
        failed_ids: set[int | str],
        state: "_EmbedRunState",
        pbar: Any,
    ) -> bool:
        """Embed one batch and persist successes. Returns True if any progress was made."""
        data_list = [data for _, data in items_to_embed]
        embeddings = asyncio.run(run_with_concurrency_limit(data_list, embed_func, max_concurrency, error_msg))
        valid_updates, batch_failed_ids = _partition_embedding_results(items_to_embed, embeddings)

        if batch_failed_ids:
            failed_ids.update(batch_failed_ids)
            logger.warning(
                f"Skipping {len(batch_failed_ids)} {display_name} that failed to embed in this run "
                f"(IDs: {batch_failed_ids[:5]}{'...' if len(batch_failed_ids) > 5 else ''})"
            )

        if valid_updates:
            ids_to_update = [item_id for item_id, _ in valid_updates]
            embeddings_to_update = [emb for _, emb in valid_updates]
            batch_count = self._set_embeddings(ids_to_update, embeddings_to_update, repo_attr, is_multi_vector)
            state.total_embedded += batch_count
            pbar.update(batch_count)
            return True

        return bool(batch_failed_ids)

    def _populate_bm25_tokens(
        self,
        tokenizer: str = "bert",
        entity_type: Literal["chunk", "query"] = "chunk",
        batch_size: int = 1000,
    ) -> int:
        """Populate BM25 tokens for all chunks or queries that don't have them.

        This is called automatically by _embed_entities when entity_type is "chunk" or "query"
        and bm25_tokenizer is specified.

        Args:
            tokenizer: Tokenizer name for BM25 sparse retrieval.
                Available tokenizers (pg_tokenizer pre-built models):
                    - "bert": bert-base-uncased (Hugging Face) - Default
                    - "wiki_tocken": Wikitext-103 trained model
                    - "gemma2b": Google lightweight model (~100MB memory)
                    - "llmlingua2": Microsoft summarization model (~200MB memory)
                See: https://github.com/tensorchord/pg_tokenizer.rs/blob/main/docs/06-model.md
            entity_type: Type of entity to populate ("chunk" or "query").
            batch_size: Number of entities to update per batch (default: 1000).

        Returns:
            Number of entities updated with BM25 tokens.
        """
        repo_attr = "chunks" if entity_type == "chunk" else "queries"

        with self._create_uow() as uow:
            repository = getattr(uow, repo_attr, None)
            if repository is None:
                raise RepositoryNotSupportedError(repo_attr, type(uow).__name__)

            try:
                # Note: batch_update_bm25_tokens commits internally per batch
                updated = repository.batch_update_bm25_tokens(tokenizer=tokenizer, batch_size=batch_size)
            except Exception as e:
                # VectorChord-BM25 extension may not be installed
                logger.warning(
                    f"Failed to generate BM25 tokens for {entity_type}s (extension may not be installed): {e}"
                )
                return 0
            else:
                logger.info(f"Generated BM25 tokens for {updated} {entity_type}s using tokenizer '{tokenizer}'")
                return updated

    def embed_all_queries(
        self,
        embed_func: TextEmbeddingFunc,
        batch_size: int = 100,
        max_concurrency: int = 10,
        bm25_tokenizer: str | None = "bert",
    ) -> int:
        """Embed all queries that don't have embeddings.

        Args:
            embed_func: Async function that takes query text and returns embedding vector.
            batch_size: Number of queries to process per batch.
            max_concurrency: Maximum concurrent embedding calls.
            bm25_tokenizer: Tokenizer for BM25 sparse retrieval. Default "bert".
                           Set to None to skip BM25 token generation.

        Returns:
            Total number of queries successfully embedded.
        """
        return self._embed_entities("query", "single", embed_func, batch_size, max_concurrency, bm25_tokenizer)

    def embed_all_queries_multi_vector(
        self,
        embed_func: TextMultiVectorEmbeddingFunc,
        batch_size: int = 100,
        max_concurrency: int = 10,
        bm25_tokenizer: str | None = "bert",
    ) -> int:
        """Embed all queries that don't have multi-vector embeddings.

        Args:
            embed_func: Async function that takes query text and returns multi-vector embedding.
            batch_size: Number of queries to process per batch.
            max_concurrency: Maximum concurrent embedding calls.
            bm25_tokenizer: Tokenizer for BM25 sparse retrieval. Default "bert".
                           Set to None to skip BM25 token generation.

        Returns:
            Total number of queries successfully embedded.
        """
        return self._embed_entities("query", "multi_vector", embed_func, batch_size, max_concurrency, bm25_tokenizer)

    def embed_all_chunks(
        self,
        embed_func: TextEmbeddingFunc,
        batch_size: int = 100,
        max_concurrency: int = 10,
        bm25_tokenizer: str | None = "bert",
    ) -> int:
        """Embed all chunks that don't have embeddings.

        Args:
            embed_func: Async function that takes chunk text and returns embedding vector.
            batch_size: Number of chunks to process per batch.
            max_concurrency: Maximum concurrent embedding calls.
            bm25_tokenizer: Tokenizer for BM25 sparse retrieval. Default "bert".
                           Set to None to skip BM25 token generation.

        Returns:
            Total number of chunks successfully embedded.
        """
        return self._embed_entities("chunk", "single", embed_func, batch_size, max_concurrency, bm25_tokenizer)

    def embed_all_chunks_multi_vector(
        self,
        embed_func: TextMultiVectorEmbeddingFunc,
        batch_size: int = 100,
        max_concurrency: int = 10,
        bm25_tokenizer: str | None = "bert",
    ) -> int:
        """Embed all chunks that don't have multi-vector embeddings.

        Args:
            embed_func: Async function that takes chunk text and returns multi-vector embedding.
            batch_size: Number of chunks to process per batch.
            max_concurrency: Maximum concurrent embedding calls.
            bm25_tokenizer: Tokenizer for BM25 sparse retrieval. Default "bert".
                           Set to None to skip BM25 token generation.

        Returns:
            Total number of chunks successfully embedded.
        """
        return self._embed_entities("chunk", "multi_vector", embed_func, batch_size, max_concurrency, bm25_tokenizer)

        # ==================== Retrieval Ground Truth Operations ====================

    def _insert_retrieval_relations(self, relations: list[dict]) -> list[tuple[int, int, int]]:
        """Internal: Insert relation dicts into database.

        This is the low-level insertion method used by all public APIs.

        Args:
            relations: List of dicts with keys:
                      - query_id (int) - required
                      - chunk_id (int | None) - FK to Chunk (mutually exclusive with image_chunk_id)
                      - image_chunk_id (int | None) - FK to ImageChunk (mutually exclusive with chunk_id)
                      - group_index (int) - required
                      - group_order (int) - required
                      - score (int | None) - optional graded relevance score

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
                    score=rel.get("score"),
                )
                for rel in relations
            ]
            uow.retrieval_relations.add_all(relation_entities)
            uow.flush()
            pks = [(r.query_id, r.group_index, r.group_order) for r in relation_entities]
            uow.commit()
            return pks

    # ==================== New Unified Retrieval GT API ====================
    # Import types from retrieval_gt module for type hints
    # These imports are done inside methods to avoid circular imports

    def add_retrieval_gt(
        self,
        query_id: int | str,
        gt: Any,
        chunk_type: Literal["mixed", "text", "image"] = "mixed",
        upsert: bool = False,
    ) -> list[tuple[int, int, int]]:
        """Add retrieval ground truth for a query.

        Args:
            query_id: The query ID.
            gt: Ground truth expression. Can be:
                - Single int (for text/image mode): 42
                - TextId/ImageId wrappers (for mixed mode): TextId(1), ImageId(2)
                - Expressions with | (OR) and & (AND) operators
                - Helper functions: or_all([1, 2, 3]) or and_all([1, 2, 3])
            chunk_type: The chunk type mode:
                - "mixed": Requires explicit TextId/ImageId wrappers
                - "text": Plain ints are treated as text chunk IDs
                - "image": Plain ints are treated as image chunk IDs
            upsert: If True, overwrite existing relations for the query.
                   If False (default), raise DuplicateRetrievalGTError if relations already exist.

        Returns:
            List of created RetrievalRelation PKs as (query_id, group_index, group_order) tuples.

        Raises:
            DuplicateRetrievalGTError: If upsert=False and relations already exist for the query.

        Examples:
            from autorag_research.orm.models import TextId, ImageId, text, image, or_all, and_all

            # Mixed mode (default) - requires explicit wrappers
            service.add_retrieval_gt(query_id=1, gt=TextId(1) | TextId(2) | ImageId(3))

            # Text mode - plain ints work
            service.add_retrieval_gt(query_id=1, gt=10, chunk_type="text")
            service.add_retrieval_gt(query_id=1, gt=or_all([1, 2, 3]), chunk_type="text")

            # Image mode
            service.add_retrieval_gt(query_id=1, gt=10, chunk_type="image")
            service.add_retrieval_gt(query_id=1, gt=image(1) | image(2), chunk_type="image")

            # Upsert mode - overwrite existing relations
            service.add_retrieval_gt(query_id=1, gt=10, chunk_type="text", upsert=True)
        """
        from autorag_research.orm.models.retrieval_gt import gt_to_relations, normalize_gt

        # Check for existing relations and handle upsert
        with self._create_uow() as uow:
            if uow.session is None:
                raise SessionNotSetError
            existing_count = uow.retrieval_relations.count_by_query(query_id)
            if existing_count > 0:
                if upsert:
                    uow.retrieval_relations.delete_by_query_id(query_id)
                    uow.commit()
                else:
                    raise DuplicateRetrievalGTError([query_id])

        normalized = normalize_gt(gt, chunk_type=chunk_type)
        relations = gt_to_relations(query_id, normalized)
        return self._insert_retrieval_relations(relations)

    def add_retrieval_gt_batch(
        self,
        items: list[tuple[int | str, Any]],
        chunk_type: Literal["mixed", "text", "image"] = "mixed",
        upsert: bool = False,
    ) -> list[tuple[int, int, int]]:
        """Batch add retrieval ground truth for multiple queries.

        Args:
            items: List of (query_id, gt_expression) tuples.
            chunk_type: The chunk type mode:
                - "mixed": Requires explicit TextId/ImageId wrappers
                - "text": Plain ints are treated as text chunk IDs
                - "image": Plain ints are treated as image chunk IDs
            upsert: If True, overwrite existing relations for the queries.
                   If False (default), raise DuplicateRetrievalGTError if relations already exist.

        Returns:
            List of created RetrievalRelation PKs as (query_id, group_index, group_order) tuples.

        Raises:
            DuplicateRetrievalGTError: If upsert=False and relations already exist for any query.

        Examples:
            from autorag_research.orm.models import TextId, ImageId, or_all, and_all

            # Mixed mode (default)
            service.add_retrieval_gt_batch([
                (1, TextId(10)),
                (2, TextId(1) | TextId(2)),
            ])

            # Text mode
            service.add_retrieval_gt_batch([
                (1, 10),
                (2, or_all([20, 21, 22])),
                (3, and_all([30, 31])),
            ], chunk_type="text")

            # Image mode
            service.add_retrieval_gt_batch([
                (1, 10),
                (2, or_all([20, 21], image)),
            ], chunk_type="image")

            # Upsert mode - overwrite existing relations
            service.add_retrieval_gt_batch([
                (1, 10),
                (2, 20),
            ], chunk_type="text", upsert=True)
        """
        from autorag_research.orm.models.retrieval_gt import gt_to_relations, normalize_gt

        query_ids = [query_id for query_id, _ in items]

        # Check for existing relations and handle upsert
        with self._create_uow() as uow:
            if uow.session is None:
                raise SessionNotSetError
            existing_query_ids = [qid for qid in query_ids if uow.retrieval_relations.count_by_query(qid) > 0]
            if existing_query_ids:
                if upsert:
                    for qid in existing_query_ids:
                        uow.retrieval_relations.delete_by_query_id(qid)
                    uow.commit()
                else:
                    raise DuplicateRetrievalGTError(existing_query_ids)

        all_relations = []
        for query_id, gt in items:
            normalized = normalize_gt(gt, chunk_type=chunk_type)
            all_relations.extend(gt_to_relations(query_id, normalized))
        return self._insert_retrieval_relations(all_relations)
