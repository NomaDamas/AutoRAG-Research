"""Text Data Ingestion Service for AutoRAG-Research.

Provides service layer for ingesting text-based data including queries,
chunks, and retrieval ground truth relations with embedding support.
"""

from collections.abc import Callable

from llama_index.core.base.embeddings.base import BaseEmbedding
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.exceptions import EmbeddingNotSetError
from autorag_research.orm.repository.text_uow import TextOnlyUnitOfWork
from autorag_research.orm.schema import Chunk, Query, RetrievalRelation


class TextDataIngestionService:
    """Service for text-only data ingestion operations.

    Provides methods for:

    - Adding queries (with optional generation_gt)
    - Adding chunks (text-only, no parent caption required)
    - Creating retrieval ground truth relations (with multi-hop support)
    - Embedding queries and chunks using LlamaIndex BaseEmbedding

    Example:
        Basic usage with queries, chunks, and retrieval ground truth:

        ```python
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from llama_index.embeddings.openai import OpenAIEmbedding

        from autorag_research.orm.service import TextDataIngestionService

        # Setup database connection
        engine = create_engine("postgresql://user:pass@localhost/dbname")
        session_factory = sessionmaker(bind=engine)

        # Initialize service with embedding model
        embedding_model = OpenAIEmbedding(model="text-embedding-3-small")
        service = TextDataIngestionService(session_factory, embedding_model)

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

        # Embed all queries and chunks without embeddings
        queries_embedded, chunks_embedded = service.embed_all_missing(
            batch_size=32
        )
        print(f"Embedded {queries_embedded} queries, {chunks_embedded} chunks")

        # Get statistics
        stats = service.get_statistics()
        print(stats)
        ```
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        embedding_model: BaseEmbedding | None = None,
    ):
        """Initialize the text data ingestion service.

        Args:
            session_factory: SQLAlchemy sessionmaker for database connections.
            embedding_model: Optional LlamaIndex BaseEmbedding model for embeddings.
        """
        self.session_factory = session_factory
        self.embedding_model = embedding_model

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
    ) -> Query:
        """Add a single query to the database.

        Args:
            query_text: The query text content.
            generation_gt: Optional list of generation ground truth answers.

        Returns:
            The created Query entity with assigned ID.
        """
        with self._create_uow() as uow:
            query = Query(query=query_text, generation_gt=generation_gt)
            uow.queries.add(query)
            uow.commit()
            # Refresh to get the ID
            uow.session.refresh(query)
            return query

    def add_queries(
        self,
        queries: list[tuple[str, list[str] | None]],
    ) -> list[Query]:
        """Add multiple queries to the database.

        Args:
            queries: List of tuples (query_text, generation_gt).
                    generation_gt can be None for each query.

        Returns:
            List of created Query entities with assigned IDs.
        """
        with self._create_uow() as uow:
            query_entities = [Query(query=query_text, generation_gt=gen_gt) for query_text, gen_gt in queries]
            uow.queries.add_all(query_entities)
            uow.commit()
            # Refresh to get IDs
            for q in query_entities:
                uow.session.refresh(q)
            return query_entities

    def add_queries_simple(
        self,
        query_texts: list[str],
    ) -> list[Query]:
        """Add multiple queries without generation ground truth.

        Args:
            query_texts: List of query text strings.

        Returns:
            List of created Query entities with assigned IDs.
        """
        with self._create_uow() as uow:
            query_entities = [Query(query=text) for text in query_texts]
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
    ) -> Chunk:
        """Add a single chunk to the database.

        Args:
            contents: The chunk text content.
            parent_caption_id: Optional parent caption ID (for document-based chunks).

        Returns:
            The created Chunk entity with assigned ID.
        """
        with self._create_uow() as uow:
            chunk = Chunk(contents=contents, parent_caption=parent_caption_id)
            uow.chunks.add(chunk)
            uow.commit()
            uow.session.refresh(chunk)
            return chunk

    def add_chunks(
        self,
        chunks: list[tuple[str, int | None]],
    ) -> list[Chunk]:
        """Add multiple chunks to the database.

        Args:
            chunks: List of tuples (contents, parent_caption_id).
                   parent_caption_id can be None for standalone chunks.

        Returns:
            List of created Chunk entities with assigned IDs.
        """
        with self._create_uow() as uow:
            chunk_entities = [
                Chunk(contents=contents, parent_caption=parent_caption_id) for contents, parent_caption_id in chunks
            ]
            uow.chunks.add_all(chunk_entities)
            uow.commit()
            for c in chunk_entities:
                uow.session.refresh(c)
            return chunk_entities

    def add_chunks_simple(
        self,
        contents_list: list[str],
    ) -> list[Chunk]:
        """Add multiple standalone chunks (no parent caption).

        This is the "chunk-only" scenario where chunks exist without
        being tied to a document/caption structure.

        Args:
            contents_list: List of chunk text contents.

        Returns:
            List of created Chunk entities with assigned IDs.
        """
        with self._create_uow() as uow:
            chunk_entities = [Chunk(contents=contents) for contents in contents_list]
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

    def set_embedding_model(self, embedding_model: BaseEmbedding) -> None:
        """Set or update the embedding model.

        Args:
            embedding_model: LlamaIndex BaseEmbedding model instance.
        """
        self.embedding_model = embedding_model

    def embed_query(self, query_id: int) -> Query | None:
        """Embed a single query and update it in the database.

        Args:
            query_id: The query ID to embed.

        Returns:
            The updated Query with embedding, None if not found.

        Raises:
            EmbeddingNotSetError: If embedding model is not set.
        """
        if self.embedding_model is None:
            raise EmbeddingNotSetError

        with self._create_uow() as uow:
            query = uow.queries.get_by_id(query_id)
            if query is None:
                return None

            # Get embedding (use query embedding for queries)
            embedding = self.embedding_model.get_query_embedding(query.query)
            query.embedding = embedding
            uow.commit()
            return query

    def embed_chunk(self, chunk_id: int) -> Chunk | None:
        """Embed a single chunk and update it in the database.

        Args:
            chunk_id: The chunk ID to embed.

        Returns:
            The updated Chunk with embedding, None if not found.

        Raises:
            EmbeddingNotSetError: If embedding model is not set.
        """
        if self.embedding_model is None:
            raise EmbeddingNotSetError

        with self._create_uow() as uow:
            chunk = uow.chunks.get_by_id(chunk_id)
            if chunk is None:
                return None

            # Get embedding (use text embedding for chunks/documents)
            embedding = self.embedding_model.get_text_embedding(chunk.contents)
            chunk.embedding = embedding
            uow.commit()
            return chunk

    def embed_queries_batch(
        self,
        query_ids: list[int],
        batch_size: int = 32,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> int:
        """Embed multiple queries in batches.

        Args:
            query_ids: List of query IDs to embed.
            batch_size: Number of queries to embed at once.
            progress_callback: Optional callback(processed, total) for progress updates.

        Returns:
            Total number of queries successfully embedded.

        Raises:
            EmbeddingNotSetError: If embedding model is not set.
        """
        if self.embedding_model is None:
            raise EmbeddingNotSetError

        total_embedded = 0
        total = len(query_ids)

        for i in range(0, len(query_ids), batch_size):
            batch_ids = query_ids[i : i + batch_size]

            with self._create_uow() as uow:
                queries = []
                for qid in batch_ids:
                    query = uow.queries.get_by_id(qid)
                    if query:
                        queries.append(query)

                if not queries:
                    continue

                # Batch embed
                texts = [q.query for q in queries]
                embeddings = self.embedding_model.get_text_embedding_batch(texts)

                for query, emb in zip(queries, embeddings, strict=True):
                    query.embedding = emb

                uow.commit()
                total_embedded += len(queries)

            if progress_callback:
                progress_callback(min(i + batch_size, total), total)

        return total_embedded

    def embed_chunks_batch(
        self,
        chunk_ids: list[int],
        batch_size: int = 32,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> int:
        """Embed multiple chunks in batches.

        Args:
            chunk_ids: List of chunk IDs to embed.
            batch_size: Number of chunks to embed at once.
            progress_callback: Optional callback(processed, total) for progress updates.

        Returns:
            Total number of chunks successfully embedded.

        Raises:
            EmbeddingNotSetError: If embedding model is not set.
        """
        if self.embedding_model is None:
            raise EmbeddingNotSetError

        total_embedded = 0
        total = len(chunk_ids)

        for i in range(0, len(chunk_ids), batch_size):
            batch_ids = chunk_ids[i : i + batch_size]

            with self._create_uow() as uow:
                chunks = []
                for cid in batch_ids:
                    chunk = uow.chunks.get_by_id(cid)
                    if chunk:
                        chunks.append(chunk)

                if not chunks:
                    continue

                # Batch embed
                texts = [c.contents for c in chunks]
                embeddings = self.embedding_model.get_text_embedding_batch(texts)

                for chunk, emb in zip(chunks, embeddings, strict=True):
                    chunk.embedding = emb

                uow.commit()
                total_embedded += len(chunks)

            if progress_callback:
                progress_callback(min(i + batch_size, total), total)

        return total_embedded

    def embed_all_queries_without_embeddings(
        self,
        batch_size: int = 32,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> int:
        """Embed all queries that don't have embeddings yet.

        Iteratively fetches and embeds queries in batches.

        Args:
            batch_size: Number of queries per batch.
            progress_callback: Optional callback(processed, total) for progress updates.

        Returns:
            Total number of queries embedded.

        Raises:
            EmbeddingNotSetError: If embedding model is not set.
        """
        if self.embedding_model is None:
            raise EmbeddingNotSetError

        total_embedded = 0

        while True:
            with self._create_uow() as uow:
                # Fetch queries without embeddings
                queries = uow.queries.get_all(limit=batch_size)
                # Filter to only those without embeddings
                queries = [q for q in queries if q.embedding is None]

                if not queries:
                    break

                # Batch embed
                texts = [q.query for q in queries]
                embeddings = self.embedding_model.get_text_embedding_batch(texts)

                for query, emb in zip(queries, embeddings, strict=True):
                    query.embedding = emb

                uow.commit()
                total_embedded += len(queries)

            if progress_callback:
                progress_callback(total_embedded, -1)  # -1 indicates unknown total

        return total_embedded

    def embed_all_chunks_without_embeddings(
        self,
        batch_size: int = 32,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> int:
        """Embed all chunks that don't have embeddings yet.

        Iteratively fetches and embeds chunks in batches.

        Args:
            batch_size: Number of chunks per batch.
            progress_callback: Optional callback(processed, total) for progress updates.

        Returns:
            Total number of chunks embedded.

        Raises:
            EmbeddingNotSetError: If embedding model is not set.
        """
        if self.embedding_model is None:
            raise EmbeddingNotSetError

        total_embedded = 0

        while True:
            with self._create_uow() as uow:
                # Use the existing method to get chunks without embeddings
                chunks = uow.chunks.get_chunks_without_embeddings(limit=batch_size)

                if not chunks:
                    break

                # Batch embed
                texts = [c.contents for c in chunks]
                embeddings = self.embedding_model.get_text_embedding_batch(texts)

                for chunk, emb in zip(chunks, embeddings, strict=True):
                    chunk.embedding = emb

                uow.commit()
                total_embedded += len(chunks)

            if progress_callback:
                progress_callback(total_embedded, -1)  # -1 indicates unknown total

        return total_embedded

    def embed_all_missing(
        self,
        batch_size: int = 32,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> tuple[int, int]:
        """Embed all queries and chunks that are missing embeddings.

        Convenience method that calls both embed_all_queries_without_embeddings
        and embed_all_chunks_without_embeddings.

        Args:
            batch_size: Number of items per batch.
            progress_callback: Optional callback(type, processed, total) for progress updates.
                              type is "queries" or "chunks".

        Returns:
            Tuple of (queries_embedded, chunks_embedded).

        Raises:
            EmbeddingNotSetError: If embedding model is not set.
        """
        query_callback = None
        chunk_callback = None

        if progress_callback:
            query_callback = lambda p, t: progress_callback("queries", p, t)
            chunk_callback = lambda p, t: progress_callback("chunks", p, t)

        queries_embedded = self.embed_all_queries_without_embeddings(
            batch_size=batch_size,
            progress_callback=query_callback,
        )
        chunks_embedded = self.embed_all_chunks_without_embeddings(
            batch_size=batch_size,
            progress_callback=chunk_callback,
        )
        return queries_embedded, chunks_embedded

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
