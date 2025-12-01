"""Text Data Ingestion Service for AutoRAG-Research.

Provides service layer for ingesting text-based data including queries,
chunks, and retrieval ground truth relations with embedding support.
"""

import logging
from typing import Any

from autorag_research.exceptions import SessionNotSetError
from autorag_research.orm.repository.text_uow import TextOnlyUnitOfWork
from autorag_research.orm.service.base_ingestion import BaseIngestionService

logger = logging.getLogger("AutoRAG-Research")


class TextDataIngestionService(BaseIngestionService):
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

    def _create_uow(self) -> TextOnlyUnitOfWork:
        """Create a new TextOnlyUnitOfWork instance.

        Returns:
            New TextOnlyUnitOfWork instance.
        """
        return TextOnlyUnitOfWork(self.session_factory, self._schema)

    def _get_schema_classes(self) -> tuple[type, type, type]:
        """Get Query, Chunk, RetrievalRelation classes from schema.

        Returns:
            Tuple of (Query, Chunk, RetrievalRelation) model classes.
        """
        if self._schema is not None:
            return self._schema.Query, self._schema.Chunk, self._schema.RetrievalRelation
        # Use default schema
        from autorag_research.orm.schema import Chunk, Query, RetrievalRelation

        return Query, Chunk, RetrievalRelation

    # ==================== Query Operations ====================

    def add_query(
        self,
        query_text: str,
        generation_gt: list[str] | None = None,
        qid: int | None = None,
    ) -> int:
        """Add a single query to the database.

        Args:
            query_text: The query text content.
            generation_gt: Optional list of generation ground truth answers.
            qid: Optional query ID to set explicitly.

        Returns:
            The created Query ID.
        """
        Query, _, _ = self._get_schema_classes()
        with self._create_uow() as uow:
            if uow.session is None:
                raise SessionNotSetError
            if qid is not None:
                query = Query(id=qid, query=query_text, generation_gt=generation_gt)
            else:
                query = Query(query=query_text, generation_gt=generation_gt)
            uow.queries.add(query)
            uow.flush()
            query_id = query.id
            uow.commit()
            return query_id

    def add_queries(
        self,
        queries: list[tuple[str, list[str] | None]],
        qids: list[int] | None = None,
    ) -> list[int]:
        """Add multiple queries to the database.

        Args:
            queries: List of tuples (query_text, generation_gt).
                    generation_gt can be None for each query.
            qids: Optional list of query IDs to set explicitly.

        Returns:
            List of created Query IDs.
        """
        Query, _, _ = self._get_schema_classes()
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
            uow.flush()
            query_ids = [q.id for q in query_entities]
            uow.commit()
            return query_ids

    def add_queries_simple(
        self,
        query_texts: list[str],
        qids: list[int] | None = None,
    ) -> list[int]:
        """Add multiple queries without generation ground truth.

        Args:
            query_texts: List of query text strings.
            qids: Optional list of query IDs to set explicitly.

        Returns:
            List of created Query IDs.
        """
        Query, _, _ = self._get_schema_classes()
        with self._create_uow() as uow:
            if uow.session is None:
                raise SessionNotSetError
            if qids is None:
                query_entities = [Query(query=text) for text in query_texts]
            else:
                query_entities = [Query(id=qid, query=text) for text, qid in zip(query_texts, qids, strict=True)]
            uow.queries.add_all(query_entities)
            uow.flush()
            query_ids = [q.id for q in query_entities]
            uow.commit()
            return query_ids

    def get_query_by_text(self, query_text: str) -> Any | None:
        """Get a query by its text content.

        Args:
            query_text: The query text to search for.

        Returns:
            The Query if found, None otherwise.
        """
        with self._create_uow() as uow:
            return uow.queries.get_by_query_text(query_text)

    def get_query_by_id(self, query_id: int) -> Any | None:
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
    ) -> int:
        """Add a single chunk to the database.

        Args:
            contents: The chunk text content.
            parent_caption_id: Optional parent caption ID (for document-based chunks).
            chunk_id: Optional chunk ID to set explicitly.

        Returns:
            The created Chunk ID.
        """
        _, Chunk, _ = self._get_schema_classes()
        with self._create_uow() as uow:
            if uow.session is None:
                raise SessionNotSetError
            if chunk_id is not None:
                chunk = Chunk(id=chunk_id, contents=contents, parent_caption=parent_caption_id)
            else:
                chunk = Chunk(contents=contents, parent_caption=parent_caption_id)
            uow.chunks.add(chunk)
            uow.flush()
            result_id = chunk.id
            uow.commit()
            return result_id

    def add_chunks(
        self,
        chunks: list[tuple[str, int | None]],
        chunk_ids: list[int] | None = None,
    ) -> list[int]:
        """Add multiple chunks to the database.

        Args:
            chunks: List of tuples (contents, parent_caption_id).
                   parent_caption_id can be None for standalone chunks.
            chunk_ids: Optional list of chunk IDs to set explicitly.

        Returns:
            List of created Chunk IDs.
        """
        _, Chunk, _ = self._get_schema_classes()
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
            uow.flush()
            result_ids = [c.id for c in chunk_entities]
            uow.commit()
            return result_ids

    def add_chunks_simple(
        self,
        contents_list: list[str],
        chunk_ids: list[int] | None = None,
    ) -> list[int]:
        """Add multiple standalone chunks (no parent caption).

        This is the "chunk-only" scenario where chunks exist without
        being tied to a document/caption structure.

        Args:
            contents_list: List of chunk text contents.
            chunk_ids: Optional list of chunk IDs to set explicitly.

        Returns:
            List of created Chunk IDs.
        """
        _, Chunk, _ = self._get_schema_classes()
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
            uow.flush()
            result_ids = [c.id for c in chunk_entities]
            uow.commit()
            return result_ids

    def get_chunk_by_id(self, chunk_id: int) -> Any | None:
        """Get a chunk by its ID.

        Args:
            chunk_id: The chunk ID.

        Returns:
            The Chunk if found, None otherwise.
        """
        with self._create_uow() as uow:
            return uow.chunks.get_by_id(chunk_id)

    def get_chunks_by_contents(self, contents: str) -> list[Any]:
        """Get chunks by exact contents match.

        Args:
            contents: The exact contents to search for.

        Returns:
            List of matching Chunk entities.
        """
        with self._create_uow() as uow:
            return uow.chunks.get_by_contents_exact(contents)

    # ==================== Retrieval GT Operations ====================
    def get_retrieval_gt_by_query(self, query_id: int) -> list[Any]:
        """Get all retrieval ground truth relations for a query.

        Args:
            query_id: The query ID.

        Returns:
            List of RetrievalRelation entities ordered by group_index and group_order.
        """
        with self._create_uow() as uow:
            return uow.retrieval_relations.get_by_query_id(query_id)

    # ==================== Cleaning Operations ====================

    def clean(self) -> dict[str, int]:
        """Delete empty queries and chunks along with their associated retrieval relations.

        This method should be called after data ingestion and before embedding to remove
        any queries or chunks with empty or whitespace-only content. It also removes
        associated retrieval relations to maintain referential integrity.

        Returns:
            Dictionary with counts of deleted queries and chunks.
        """
        deleted_queries = self._delete_empty_queries()
        deleted_chunks = self._delete_empty_chunks()
        return {
            "deleted_queries": deleted_queries,
            "deleted_chunks": deleted_chunks,
        }

    def _delete_empty_queries(self) -> int:
        """Delete queries with empty content and their associated retrieval relations.

        This method finds all queries where the query text is empty or whitespace-only,
        deletes their associated retrieval relations first (to avoid FK violations),
        then deletes the queries themselves.

        Returns:
            Total number of queries deleted.
        """
        total_deleted = 0

        while True:
            with self._create_uow() as uow:
                if uow.session is None:
                    raise SessionNotSetError

                # Get batch of empty queries
                empty_queries = uow.queries.get_queries_with_empty_content(limit=100)
                if not empty_queries:
                    break

                # Delete retrieval relations and queries
                for query in empty_queries:
                    # Delete retrieval relations first (FK constraint)
                    deleted_relations = uow.retrieval_relations.delete_by_query_id(query.id)
                    if deleted_relations > 0:
                        logger.info(f"Deleted {deleted_relations} retrieval relations for empty query {query.id}.")

                    # Delete the query
                    uow.queries.delete(query)
                    total_deleted += 1

                uow.commit()

        if total_deleted > 0:
            logger.warning(f"Deleted {total_deleted} queries with empty content.")

        return total_deleted

    def _delete_empty_chunks(self) -> int:
        """Delete chunks with empty content and their associated retrieval relations.

        This method finds all chunks where the contents is empty or whitespace-only,
        deletes their associated retrieval relations first (to avoid FK violations),
        then deletes the chunks themselves.

        Returns:
            Total number of chunks deleted.
        """
        total_deleted = 0

        while True:
            with self._create_uow() as uow:
                if uow.session is None:
                    raise SessionNotSetError

                # Get batch of empty chunks
                empty_chunks = uow.chunks.get_chunks_with_empty_content(limit=100)
                if not empty_chunks:
                    break

                # Delete retrieval relations and chunks
                for chunk in empty_chunks:
                    # Delete retrieval relations first (FK constraint)
                    deleted_relations = uow.retrieval_relations.delete_by_chunk_id(chunk.id)
                    if deleted_relations > 0:
                        logger.info(f"Deleted {deleted_relations} retrieval relations for empty chunk {chunk.id}.")

                    # Delete the chunk
                    uow.chunks.delete(chunk)
                    total_deleted += 1

                uow.commit()

        if total_deleted > 0:
            logger.warning(f"Deleted {total_deleted} chunks with empty content.")

        return total_deleted

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
            chunks_with_emb = len(uow.chunks.get_with_embeddings())
            chunks_without_emb = len(uow.chunks.get_without_embeddings())

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
