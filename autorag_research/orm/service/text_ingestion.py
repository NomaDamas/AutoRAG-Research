"""Text Data Ingestion Service for AutoRAG-Research.

Provides service layer for ingesting text-based data including queries,
chunks, and retrieval ground truth relations with embedding support.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable

from sqlalchemy.orm import Session, sessionmaker

from autorag_research.exceptions import LengthMismatchError, SessionNotSetError
from autorag_research.orm.repository.text_uow import TextOnlyUnitOfWork
from autorag_research.orm.schema import Chunk, Query, RetrievalRelation

EmbeddingFunc = Callable[[str], Awaitable[list[float]]]
logger = logging.getLogger("AutoRAG-Research")


class TextDataIngestionService:
    """Service for text-only data ingestion operations.

    Provides methods for:

    - Adding queries (with optional generation_gt)
    - Adding chunks (text-only, no parent caption required)
    - Creating retrieval ground truth relations (with multi-hop support)
    - Setting embeddings for queries and chunks (accepts pre-computed vectors)

    Example:
        Basic usage with queries, chunks, and retrieval ground truth:

        ```python
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from autorag_research.orm.service import TextDataIngestionService

        # Setup database connection
        engine = create_engine("postgresql://user:pass@localhost/dbname")
        session_factory = sessionmaker(bind=engine)

        # Initialize service
        service = TextDataIngestionService(session_factory)

        # Add queries with generation ground truth
        query = service.add_query(
            query_text="What is RAG?",
            generation_gt=["Retrieval Augmented Generation"]
        )

        # Add standalone chunks (chunk-only scenario)
        chunks = service.add_chunks_simple([
            "RAG combines retrieval with generation...",
            "The retrieval component fetches relevant documents...",
        ])

        # Add retrieval ground truth (non-multi-hop)
        service.add_retrieval_gt_simple(
            query_id=query.id,
            chunk_ids=[c.id for c in chunks]
        )

        # For multi-hop scenarios (different hops need different groups)
        service.add_retrieval_gt_multihop(
            query_id=query.id,
            chunk_groups=[
                [chunks[0].id],  # First hop
                [chunks[1].id],  # Second hop
            ]
        )

        # Set embeddings using pre-computed vectors
        query_embedding = [0.1, 0.2, ...]  # from your embedding model
        service.set_query_embedding(query.id, query_embedding)

        chunk_embeddings = [[0.1, 0.2, ...], [0.3, 0.4, ...]]  # from your embedding model
        service.set_chunk_embeddings(
            [c.id for c in chunks],
            chunk_embeddings
        )

        # Get statistics
        stats = service.get_statistics()
        print(stats)
        ```
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session],
    ):
        """Initialize the text data ingestion service.

        Args:
            session_factory: SQLAlchemy sessionmaker for database connections.
        """
        self.session_factory = session_factory

    def _create_uow(self) -> TextOnlyUnitOfWork:
        """Create a new TextOnlyUnitOfWork instance.

        Returns:
            New TextOnlyUnitOfWork instance.
        """
        return TextOnlyUnitOfWork(self.session_factory)

    # ==================== Query Operations ====================

    def add_query(
        self,
        query_text: str,
        generation_gt: list[str] | None = None,
        qid: int | None = None,
    ) -> Query:
        """Add a single query to the database.

        Args:
            query_text: The query text content.
            generation_gt: Optional list of generation ground truth answers.
            qid: Optional query ID to set explicitly.

        Returns:
            The created Query entity with assigned ID.
        """
        with self._create_uow() as uow:
            if uow.session is None:
                raise SessionNotSetError
            if qid is not None:
                query = Query(id=qid, query=query_text, generation_gt=generation_gt)
            else:
                query = Query(query=query_text, generation_gt=generation_gt)
            uow.queries.add(query)
            uow.commit()
            # Refresh to get the ID
            uow.session.refresh(query)
            return query

    def add_queries(
        self,
        queries: list[tuple[str, list[str] | None]],
        qids: list[int] | None = None,
    ) -> list[Query]:
        """Add multiple queries to the database.

        Args:
            queries: List of tuples (query_text, generation_gt).
                    generation_gt can be None for each query.
            qids: Optional list of query IDs to set explicitly.

        Returns:
            List of created Query entities with assigned IDs.
        """
        with self._create_uow() as uow:
            if uow.session is None:
                raise SessionNotSetError
            if qids is None:
                query_entities = [Query(query=query_text, generation_gt=gen_gt) for query_text, gen_gt in queries]
            else:
                query_entities = [
                    Query(id=qid, query=query_text, generation_gt=gen_gt)
                    for (query_text, gen_gt), qid in zip(queries, qids, strict=True)
                ]
            uow.queries.add_all(query_entities)
            uow.commit()
            # Refresh to get IDs
            for q in query_entities:
                uow.session.refresh(q)
            return query_entities

    def add_queries_simple(
        self,
        query_texts: list[str],
        qids: list[int] | None = None,
    ) -> list[Query]:
        """Add multiple queries without generation ground truth.

        Args:
            query_texts: List of query text strings.
            qids: Optional list of query IDs to set explicitly.

        Returns:
            List of created Query entities with assigned IDs.
        """
        with self._create_uow() as uow:
            if uow.session is None:
                raise SessionNotSetError
            if qids is None:
                query_entities = [Query(query=text) for text in query_texts]
            else:
                query_entities = [Query(id=qid, query=text) for text, qid in zip(query_texts, qids, strict=True)]
            uow.queries.add_all(query_entities)
            uow.commit()
            for q in query_entities:
                uow.session.refresh(q)
            return query_entities

    def get_query_by_text(self, query_text: str) -> Query | None:
        """Get a query by its text content.

        Args:
            query_text: The query text to search for.

        Returns:
            The Query if found, None otherwise.
        """
        with self._create_uow() as uow:
            return uow.queries.get_by_query_text(query_text)

    def get_query_by_id(self, query_id: int) -> Query | None:
        """Get a query by its ID.

        Args:
            query_id: The query ID.

        Returns:
            The Query if found, None otherwise.
        """
        with self._create_uow() as uow:
            return uow.queries.get_by_id(query_id)

    # ==================== Chunk Operations ====================

    def add_chunk(
        self,
        contents: str,
        parent_caption_id: int | None = None,
        chunk_id: int | None = None,
    ) -> Chunk:
        """Add a single chunk to the database.

        Args:
            contents: The chunk text content.
            parent_caption_id: Optional parent caption ID (for document-based chunks).
            chunk_id: Optional chunk ID to set explicitly.

        Returns:
            The created Chunk entity with assigned ID.
        """
        with self._create_uow() as uow:
            if uow.session is None:
                raise SessionNotSetError
            if chunk_id is not None:
                chunk = Chunk(id=chunk_id, contents=contents, parent_caption=parent_caption_id)
            else:
                chunk = Chunk(contents=contents, parent_caption=parent_caption_id)
            uow.chunks.add(chunk)
            uow.commit()
            uow.session.refresh(chunk)
            return chunk

    def add_chunks(
        self,
        chunks: list[tuple[str, int | None]],
        chunk_ids: list[int] | None = None,
    ) -> list[Chunk]:
        """Add multiple chunks to the database.

        Args:
            chunks: List of tuples (contents, parent_caption_id).
                   parent_caption_id can be None for standalone chunks.
            chunk_ids: Optional list of chunk IDs to set explicitly.

        Returns:
            List of created Chunk entities with assigned IDs.
        """
        with self._create_uow() as uow:
            if uow.session is None:
                raise SessionNotSetError
            if chunk_ids is None:
                chunk_entities = [
                    Chunk(contents=contents, parent_caption=parent_caption_id) for contents, parent_caption_id in chunks
                ]
            else:
                chunk_entities = [
                    Chunk(id=cid, contents=contents, parent_caption=parent_caption_id)
                    for (contents, parent_caption_id), cid in zip(chunks, chunk_ids, strict=True)
                ]
            uow.chunks.add_all(chunk_entities)
            uow.commit()
            for c in chunk_entities:
                uow.session.refresh(c)
            return chunk_entities

    def add_chunks_simple(
        self,
        contents_list: list[str],
        chunk_ids: list[int] | None = None,
    ) -> list[Chunk]:
        """Add multiple standalone chunks (no parent caption).

        This is the "chunk-only" scenario where chunks exist without
        being tied to a document/caption structure.

        Args:
            contents_list: List of chunk text contents.
            chunk_ids: Optional list of chunk IDs to set explicitly.

        Returns:
            List of created Chunk entities with assigned IDs.
        """
        with self._create_uow() as uow:
            if uow.session is None:
                raise SessionNotSetError
            if chunk_ids is None:
                chunk_entities = [Chunk(contents=contents) for contents in contents_list]
            else:
                chunk_entities = [
                    Chunk(id=cid, contents=contents) for contents, cid in zip(contents_list, chunk_ids, strict=True)
                ]
            uow.chunks.add_all(chunk_entities)
            uow.commit()
            for c in chunk_entities:
                uow.session.refresh(c)
            return chunk_entities

    def get_chunk_by_id(self, chunk_id: int) -> Chunk | None:
        """Get a chunk by its ID.

        Args:
            chunk_id: The chunk ID.

        Returns:
            The Chunk if found, None otherwise.
        """
        with self._create_uow() as uow:
            return uow.chunks.get_by_id(chunk_id)

    def get_chunks_by_contents(self, contents: str) -> list[Chunk]:
        """Get chunks by exact contents match.

        Args:
            contents: The exact contents to search for.

        Returns:
            List of matching Chunk entities.
        """
        with self._create_uow() as uow:
            return uow.chunks.get_by_contents_exact(contents)

    # ==================== Retrieval GT Operations ====================

    def add_retrieval_gt(
        self,
        query_id: int,
        chunk_id: int,
        group_index: int | None = None,
        group_order: int | None = None,
    ) -> RetrievalRelation:
        """Add a single retrieval ground truth relation.

        For non-multi-hop scenarios, group_index defaults to 0 and
        group_order auto-increments based on existing relations.

        Args:
            query_id: The query ID.
            chunk_id: The chunk ID (text chunk, not image chunk).
            group_index: Optional group index (for multi-hop, different groups).
            group_order: Optional order within group.

        Returns:
            The created RetrievalRelation entity.
        """
        with self._create_uow() as uow:
            if uow.session is None:
                raise SessionNotSetError
            # If group_index not provided, default to 0 (single-hop)
            if group_index is None:
                group_index = 0

            # If group_order not provided, auto-increment
            if group_order is None:
                max_order = uow.retrieval_relations.get_max_group_order(query_id, group_index)
                group_order = (max_order or -1) + 1

            relation = RetrievalRelation(
                query_id=query_id,
                chunk_id=chunk_id,
                group_index=group_index,
                group_order=group_order,
            )
            uow.retrieval_relations.add(relation)
            uow.commit()
            uow.session.refresh(relation)
            return relation

    def add_retrieval_gt_simple(
        self,
        query_id: int,
        chunk_ids: list[int],
    ) -> list[RetrievalRelation]:
        """Add multiple retrieval GTs for a query (non-multi-hop).

        All chunks are added to the same group (group_index=0) with
        incrementing group_order values. Use this when multi-hop
        is NOT needed.

        Args:
            query_id: The query ID.
            chunk_ids: List of chunk IDs.

        Returns:
            List of created RetrievalRelation entities.
        """
        with self._create_uow() as uow:
            if uow.session is None:
                raise SessionNotSetError
            # Get current max order for group 0
            max_order = uow.retrieval_relations.get_max_group_order(query_id, 0)
            start_order = (max_order or -1) + 1

            relations = [
                RetrievalRelation(
                    query_id=query_id,
                    chunk_id=chunk_id,
                    group_index=0,
                    group_order=start_order + i,
                )
                for i, chunk_id in enumerate(chunk_ids)
            ]
            uow.retrieval_relations.add_all(relations)
            uow.commit()
            for r in relations:
                uow.session.refresh(r)
            return relations

    def add_retrieval_gt_multihop(
        self,
        query_id: int,
        chunk_groups: list[list[int]],
    ) -> list[RetrievalRelation]:
        """Add multiple retrieval GTs for a query with multi-hop support.

        Each inner list represents a separate "hop" or alternative path.
        Chunks in different groups have different group_index values.

        For example:
        - [[1, 2], [3, 4]] means:
          - Group 0: chunks 1, 2 (first hop)
          - Group 1: chunks 3, 4 (second hop)

        Args:
            query_id: The query ID.
            chunk_groups: List of lists of chunk IDs.
                         Each inner list is a separate group.

        Returns:
            List of all created RetrievalRelation entities.
        """
        with self._create_uow() as uow:
            if uow.session is None:
                raise SessionNotSetError
            # Get current max group index
            max_group_idx = uow.retrieval_relations.get_max_group_index(query_id)
            start_group_idx = (max_group_idx or -1) + 1

            all_relations = []
            for group_offset, chunk_ids in enumerate(chunk_groups):
                group_index = start_group_idx + group_offset
                for order, chunk_id in enumerate(chunk_ids):
                    relation = RetrievalRelation(
                        query_id=query_id,
                        chunk_id=chunk_id,
                        group_index=group_index,
                        group_order=order,
                    )
                    all_relations.append(relation)

            uow.retrieval_relations.add_all(all_relations)
            uow.commit()
            for r in all_relations:
                uow.session.refresh(r)
            return all_relations

    def get_retrieval_gt_by_query(self, query_id: int) -> list[RetrievalRelation]:
        """Get all retrieval ground truth relations for a query.

        Args:
            query_id: The query ID.

        Returns:
            List of RetrievalRelation entities ordered by group_index and group_order.
        """
        with self._create_uow() as uow:
            return uow.retrieval_relations.get_by_query_id(query_id)

    # ==================== Embedding Operations ====================

    def set_query_embedding(self, query_id: int, embedding: list[float]) -> Query | None:
        """Set the embedding for a single query.

        Args:
            query_id: The query ID to set embedding for.
            embedding: The pre-computed embedding vector.

        Returns:
            The updated Query with embedding, None if not found.
        """
        with self._create_uow() as uow:
            if uow.session is None:
                raise SessionNotSetError
            query = uow.queries.get_by_id(query_id)
            if query is None:
                return None

            query.embedding = embedding
            uow.commit()
            uow.session.refresh(query)
            return query

    def set_chunk_embedding(self, chunk_id: int, embedding: list[float]) -> Chunk | None:
        """Set the embedding for a single chunk.

        Args:
            chunk_id: The chunk ID to set embedding for.
            embedding: The pre-computed embedding vector.

        Returns:
            The updated Chunk with embedding, None if not found.
        """
        with self._create_uow() as uow:
            if uow.session is None:
                raise SessionNotSetError
            chunk = uow.chunks.get_by_id(chunk_id)
            if chunk is None:
                return None

            chunk.embedding = embedding
            uow.commit()
            uow.session.refresh(chunk)
            return chunk

    def set_query_embeddings(
        self,
        query_ids: list[int],
        embeddings: list[list[float]],
    ) -> int:
        """Set embeddings for multiple queries.

        Args:
            query_ids: List of query IDs to set embeddings for.
            embeddings: List of pre-computed embedding vectors (must match query_ids length).

        Returns:
            Total number of queries successfully updated.

        Raises:
            ValueError: If query_ids and embeddings have different lengths.
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
            embeddings: List of pre-computed embedding vectors (must match chunk_ids length).

        Returns:
            Total number of chunks successfully updated.

        Raises:
            ValueError: If chunk_ids and embeddings have different lengths.
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

        Processes queries in batches, using semaphore to limit concurrent
        embedding calls. After each batch completes, updates the database.

        Args:
            embed_func: Async function that takes a text string and returns embedding vector.
            batch_size: Number of queries to process per batch before DB update.
            max_concurrency: Maximum number of concurrent embedding calls.

        Returns:
            Total number of queries successfully embedded.
        """
        total_embedded = 0

        while True:
            # Get queries without embeddings for this batch
            with self._create_uow() as uow:
                queries = uow.queries.get_queries_without_embeddings(limit=batch_size)
                if not queries:
                    break
                items_to_embed = [(q.id, q.query) for q in queries]

            # Embed batch with semaphore
            embeddings = asyncio.run(self._embed_batch(items_to_embed, embed_func, max_concurrency))

            # Update database with embeddings
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

        Processes chunks in batches, using semaphore to limit concurrent
        embedding calls. After each batch completes, updates the database.

        Args:
            embed_func: Async function that takes a text string and returns embedding vector.
            batch_size: Number of chunks to process per batch before DB update.
            max_concurrency: Maximum number of concurrent embedding calls.

        Returns:
            Total number of chunks successfully embedded.
        """
        total_embedded = 0

        while True:
            # Get chunks without embeddings for this batch
            with self._create_uow() as uow:
                chunks = uow.chunks.get_chunks_without_embeddings(limit=batch_size)
                if not chunks:
                    break
                items_to_embed = [(c.id, c.contents) for c in chunks]

            # Embed batch with semaphore
            embeddings = asyncio.run(self._embed_batch(items_to_embed, embed_func, max_concurrency))

            # Update database with embeddings
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
    async def _embed_batch(
        items: list[tuple[int, str]],
        embed_func: EmbeddingFunc,
        max_concurrency: int,
    ) -> list[list[float] | None]:
        """Embed a batch of items with concurrency control.

        Args:
            items: List of (id, text) tuples to embed.
            embed_func: Async function that takes text and returns embedding.
            max_concurrency: Maximum concurrent embedding calls.

        Returns:
            List of embeddings (or None if failed) in same order as items.
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def embed_with_semaphore(text: str) -> list[float] | None:
            async with semaphore:
                try:
                    return await embed_func(text)
                except Exception:
                    logger.exception(f"Failed to embed {text}")
                    return None

        tasks = [embed_with_semaphore(text) for _, text in items]
        return await asyncio.gather(*tasks)

    # ==================== Statistics ====================

    def get_statistics(self) -> dict:
        """Get statistics about the ingested data.

        Returns:
            Dictionary with counts of queries, chunks, and embeddings status.
        """
        with self._create_uow() as uow:
            total_queries = uow.queries.count()
            total_chunks = uow.chunks.count()

            # Count queries/chunks with embeddings
            # queries_with_emb = len(uow.queries.get_all(limit=None))  # TODO: add count method
            chunks_with_emb = len(uow.chunks.get_chunks_with_embeddings())
            chunks_without_emb = len(uow.chunks.get_chunks_without_embeddings())

            return {
                "queries": {
                    "total": total_queries,
                },
                "chunks": {
                    "total": total_chunks,
                    "with_embeddings": chunks_with_emb,
                    "without_embeddings": chunks_without_emb,
                },
            }
