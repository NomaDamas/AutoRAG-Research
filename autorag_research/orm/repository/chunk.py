"""Chunk repository for AutoRAG-Research.

Implements chunk-specific CRUD operations and queries extending
the base vector repository pattern for similarity search.
"""

from typing import Any

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session, joinedload

from autorag_research.orm.repository.base import BaseEmbeddingRepository, BaseVectorRepository


class ChunkRepository(BaseVectorRepository[Any], BaseEmbeddingRepository[Any]):
    """Repository for Chunk entity with vector search capabilities."""

    def __init__(self, session: Session, model_cls: type | None = None):
        """Initialize chunk repository.

        Args:
            session: SQLAlchemy session for database operations.
            model_cls: The Chunk model class to use. If None, uses default schema.
        """
        if model_cls is None:
            from autorag_research.orm.schema import Chunk

            model_cls = Chunk
        super().__init__(session, model_cls)

    def get_with_page_chunk_relations(self, chunk_id: int | str) -> Any | None:
        """Retrieve a chunk with its page-chunk relations eagerly loaded.

        Args:
            chunk_id: The chunk ID.

        Returns:
            The chunk with page-chunk relations loaded, None if not found.
        """
        stmt = (
            select(self.model_cls)
            .where(self.model_cls.id == chunk_id)
            .options(joinedload(self.model_cls.page_chunk_relations))
        )
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def get_with_retrieval_relations(self, chunk_id: int | str) -> Any | None:
        """Retrieve a chunk with its retrieval relations eagerly loaded.

        Args:
            chunk_id: The chunk ID.

        Returns:
            The chunk with retrieval relations loaded, None if not found.
        """
        stmt = (
            select(self.model_cls)
            .where(self.model_cls.id == chunk_id)
            .options(joinedload(self.model_cls.retrieval_relations))
        )
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def get_with_chunk_retrieved_results(self, chunk_id: int | str) -> Any | None:
        """Retrieve a chunk with its chunk retrieved results eagerly loaded.

        Args:
            chunk_id: The chunk ID.

        Returns:
            The chunk with chunk retrieved results loaded, None if not found.
        """
        stmt = (
            select(self.model_cls)
            .where(self.model_cls.id == chunk_id)
            .options(joinedload(self.model_cls.chunk_retrieved_results))
        )
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def search_by_contents(self, search_text: str) -> list[Any]:
        """Search chunks by contents using SQL LIKE.

        Args:
            search_text: The text to search for (use % as wildcard).

        Returns:
            List of matching chunks.
        """
        stmt = select(self.model_cls).where(self.model_cls.contents.like(f"%{search_text}%"))
        return list(self.session.execute(stmt).scalars().all())

    def get_by_contents_exact(self, contents: str) -> list[Any]:
        """Retrieve chunks with exact contents match.

        Args:
            contents: The exact contents to search for.

        Returns:
            List of chunks with matching contents.
        """
        stmt = select(self.model_cls).where(self.model_cls.contents == contents)
        return list(self.session.execute(stmt).scalars().all())

    def get_with_all_relations(self, chunk_id: int | str) -> Any | None:
        """Retrieve a chunk with all relationships eagerly loaded.

        Args:
            chunk_id: The chunk ID.

        Returns:
            The chunk with all relations loaded, None if not found.
        """
        stmt = (
            select(self.model_cls)
            .where(self.model_cls.id == chunk_id)
            .options(
                joinedload(self.model_cls.page_chunk_relations),
                joinedload(self.model_cls.retrieval_relations),
                joinedload(self.model_cls.chunk_retrieved_results),
            )
        )
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def get_chunks_with_empty_content(self, limit: int | None = None) -> list[Any]:
        """Retrieve chunks that have empty or whitespace-only contents.

        Args:
            limit: Maximum number of results to return.

        Returns:
            List of chunks with empty content.
        """
        # Use SQL TRIM to check for empty or whitespace-only content
        stmt = select(self.model_cls).where(
            (self.model_cls.contents.is_(None)) | (func.trim(self.model_cls.contents) == "")
        )
        if limit:
            stmt = stmt.limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def get_table_chunks(self) -> list[Any]:
        """Retrieve chunks that are tables (is_table=True).

        Returns:
            List of chunks where is_table is True.
        """
        stmt = select(self.model_cls).where(self.model_cls.is_table.is_(True))
        return list(self.session.execute(stmt).scalars().all())

    def get_by_table_type(self, table_type: str) -> list[Any]:
        """Retrieve chunks with a specific table_type.

        Args:
            table_type: The table format type (e.g., 'markdown', 'xml', 'html').

        Returns:
            List of chunks with the specified table_type.
        """
        stmt = select(self.model_cls).where(self.model_cls.table_type == table_type)
        return list(self.session.execute(stmt).scalars().all())

    def get_non_table_chunks(self) -> list[Any]:
        """Retrieve chunks that are not tables (is_table=False).

        Returns:
            List of chunks where is_table is False.
        """
        stmt = select(self.model_cls).where(self.model_cls.is_table.is_(False))
        return list(self.session.execute(stmt).scalars().all())

    def bm25_search(
        self,
        query_text: str,
        index_name: str = "idx_chunk_bm25",
        limit: int = 10,
        tokenizer: str = "bert",
    ) -> list[tuple[Any, float]]:
        """Perform VectorChord BM25 search.

        Uses VectorChord-BM25's <&> operator for full-text search.
        Returns entities with their BM25 scores (converted to positive values).

        Args:
            query_text: The query text to search for.
            index_name: Name of the BM25 index (default: "idx_chunk_bm25").
            limit: Maximum number of results to return.
            tokenizer: Tokenizer to use for query (default: "bert").
                Available tokenizers (pg_tokenizer pre-built models):
                    - "bert": bert-base-uncased (Hugging Face) - Default
                    - "wiki_tocken": Wikitext-103 trained model
                    - "gemma2b": Google lightweight model (~100MB memory)
                    - "llmlingua2": Microsoft summarization model (~200MB memory)
                See: https://github.com/tensorchord/pg_tokenizer.rs/blob/main/docs/06-model.md

        Returns:
            List of tuples (entity, score) ordered by relevance.
            Higher scores indicate higher relevance.

        Note:
            BM25 scores from VectorChord are negative (more negative = more relevant).
            This method negates the scores so higher = more relevant.
        """
        table_name = self.model_cls.__tablename__

        # BM25 search query using <&> operator with to_bm25query
        sql = text(f"""
            SELECT id,
                   bm25_tokens <&> to_bm25query(:index_name, tokenize(:query, :tokenizer)::bm25vector) AS score
            FROM {table_name}
            WHERE bm25_tokens IS NOT NULL
            ORDER BY score
            LIMIT :limit
        """)  # noqa: S608

        result = self.session.execute(
            sql,
            {"index_name": index_name, "query": query_text, "tokenizer": tokenizer, "limit": limit},
        )
        rows = result.fetchall()

        # Build id -> score mapping (negate scores so higher = better)
        id_to_score = {row[0]: -float(row[1]) for row in rows}

        # Fetch all entities in single query
        entities = self.get_by_ids(list(id_to_score.keys()))
        id_to_entity = {entity.id: entity for entity in entities}

        # Return entities with scores, sorted by score descending (higher = better)
        return sorted(
            [(id_to_entity[eid], score) for eid, score in id_to_score.items() if eid in id_to_entity],
            key=lambda x: x[1],
            reverse=True,
        )
