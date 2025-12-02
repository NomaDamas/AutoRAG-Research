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

    def _get_schema_classes(self) -> dict[str, type]:
        """Get Query, Chunk, RetrievalRelation classes from schema.

        Returns:
            Tuple of (Query, Chunk, RetrievalRelation) model classes.
        """
        if self._schema is not None:
            return {
                "Query": self._schema.Query,
                "Chunk": self._schema.Chunk,
                "RetrievalRelation": self._schema.RetrievalRelation,
            }
        # Use default schema
        from autorag_research.orm.schema import Chunk, Query, RetrievalRelation

        return {
            "Query": Query,
            "Chunk": Chunk,
            "RetrievalRelation": RetrievalRelation,
        }

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
