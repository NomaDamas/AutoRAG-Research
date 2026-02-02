"""Repository layer for AutoRAG-Research.

Implements Generic Repository + Unit of Work patterns for efficient
CRUD operations and transaction management with SQLAlchemy.
"""

import logging
from contextlib import contextmanager
from typing import Any, Generic, TypeVar, cast

from sqlalchemy import CursorResult, select, text
from sqlalchemy.orm import Session

from autorag_research.exceptions import LengthMismatchError, NoSessionError

T = TypeVar("T")

logger = logging.getLogger("AutoRAG-Research")


def _vec_to_pg_literal(vec: list[float]) -> str:
    """Convert a vector to PostgreSQL literal format.

    Args:
        vec: List of floats representing a vector.

    Returns:
        PostgreSQL vector literal string, e.g., "[0.1,0.2,0.3]"
    """
    return "[" + ",".join(str(float(x)) for x in vec) + "]"


def _vecs_to_pg_array(vecs: list[list[float]]) -> str:
    """Convert multiple vectors to PostgreSQL array literal format.

    Args:
        vecs: List of vectors (list of lists of floats).

    Returns:
        PostgreSQL array of vectors literal, e.g., "ARRAY['[0.1,0.2]'::vector, '[0.3,0.4]'::vector]"
    """
    vector_literals = [f"'{_vec_to_pg_literal(vec)}'::vector" for vec in vecs]
    return f"ARRAY[{','.join(vector_literals)}]"


class GenericRepository(Generic[T]):
    """Generic repository implementing common CRUD operations.

    This base class provides reusable database operations that can be
    extended by specific repositories for custom business logic.
    """

    def __init__(self, session: Session, model_cls: type[T]):
        """Initialize repository with a session and model class.

        Args:
            session: SQLAlchemy session for database operations.
            model_cls: The SQLAlchemy model class this repository manages.
        """
        self.session = session
        self.model_cls = model_cls

    def add(self, entity: T) -> T:
        """Add a new entity to the session.

        Args:
            entity: The entity instance to add.

        Returns:
            The added entity.
        """
        self.session.add(entity)
        return entity

    def add_all(self, entities: list[T]) -> list[T]:
        """Add multiple entities to the session.

        Args:
            entities: List of entity instances to add.

        Returns:
            The added entities.
        """
        self.session.add_all(entities)
        return entities

    def add_bulk(self, items: list[dict]) -> list[Any]:
        """Memory-efficient bulk insert using SQLAlchemy Core.

        Unlike add_all(), this method does not create ORM objects in Python memory.
        Instead, it uses SQLAlchemy Core's insert() which generates a single
        multi-row INSERT statement, significantly reducing memory usage and
        improving performance for large batch inserts.

        Args:
            items: List of dictionaries representing records to insert.

        Returns:
            List of inserted IDs.

        Note:
            For 1000 records, this method uses ~3-5x less memory than add_all()
            because it bypasses ORM object creation and identity map tracking.
        """
        from sqlalchemy import insert

        if not items:
            return []

        stmt = insert(self.model_cls).values(items).returning(self.model_cls.id)  # ty: ignore[possibly-missing-attribute]
        result = self.session.execute(stmt)
        return [row[0] for row in result]

    def get_by_id(self, _id: Any) -> T | None:
        """Retrieve an entity by its primary key.

        Args:
            _id: The primary key value.

        Returns:
            The entity if found, None otherwise.
        """
        return self.session.get(self.model_cls, _id)

    def get_by_ids(self, ids: list[Any]) -> list[T]:
        """Retrieve multiple entities by their primary keys.

        Args:
            ids: List of primary key values.

        Returns:
            List of entities found (may be fewer than requested if some don't exist).
        """
        if not ids:
            return []
        # Assume 'id' is the primary key column name
        stmt = select(self.model_cls).where(self.model_cls.id.in_(ids))  # ty: ignore[possibly-missing-attribute]
        return list(self.session.execute(stmt).scalars().all())

    def get_all(self, limit: int | None = None, offset: int | None = None) -> list[T]:
        """Retrieve all entities of this type.

        Args:
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of all entities.
        """
        stmt = select(self.model_cls)
        if offset:
            stmt = stmt.offset(offset)
        if limit:
            stmt = stmt.limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def update(self, entity: T) -> T:
        """Update an existing entity.

        Args:
            entity: The entity instance to update.

        Returns:
            The updated entity.
        """
        return self.session.merge(entity)

    def delete(self, entity: T) -> None:
        """Delete an entity from the database.

        Args:
            entity: The entity instance to delete.
        """
        self.session.delete(entity)

    def delete_by_id(self, _id: Any) -> bool:
        """Delete an entity by its primary key.

        Args:
            _id: The primary key value.

        Returns:
            True if entity was deleted, False if not found.
        """
        entity = self.get_by_id(_id)
        if entity:
            self.delete(entity)
            return True
        return False

    def count(self) -> int:
        """Count total number of entities.

        Returns:
            Total count of entities.
        """
        return self.session.query(self.model_cls).count()

    def exists(self, _id: Any) -> bool:
        """Check if an entity exists by its primary key.

        Args:
            _id: The primary key value.

        Returns:
            True if entity exists, False otherwise.
        """
        return self.get_by_id(_id) is not None


class UnitOfWork:
    """Unit of Work pattern for managing database transactions.

    Ensures data consistency by grouping multiple repository operations
    into a single atomic transaction.
    """

    def __init__(self, session_factory: Any):
        """Initialize Unit of Work with a session factory.

        Args:
            session_factory: SQLAlchemy sessionmaker instance.
        """
        self.session_factory = session_factory
        self.session: Session | None = None

    def __enter__(self) -> "UnitOfWork":
        """Enter the context manager and create a new session.

        Returns:
            Self for method chaining.
        """
        self.session = self.session_factory()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context manager and clean up session.

        Automatically rolls back if an exception occurred.

        Args:
            exc_type: Exception type if an error occurred.
            exc_val: Exception value if an error occurred.
            exc_tb: Exception traceback if an error occurred.
        """
        if exc_type is not None:
            self.rollback()
        if self.session:
            self.session.close()

    def commit(self) -> None:
        """Commit the current transaction."""
        if self.session:
            self.session.commit()

    def rollback(self) -> None:
        """Rollback the current transaction."""
        if self.session:
            self.session.rollback()

    def flush(self) -> None:
        """Flush pending changes without committing."""
        if self.session:
            self.session.flush()


class BaseVectorRepository(GenericRepository[T]):
    """Base repository with vector search capabilities.

    Extends GenericRepository with vector search methods for use with
    pgvector and VectorChord for efficient similarity search.
    """

    def vector_search(
        self,
        query_vector: list[float],
        vector_column: str = "embedding",
        limit: int = 10,
    ) -> list[T]:
        """Perform vector similarity search using VectorChord's cosine distance.

        Uses raw SQL with VectorChord's <=> operator for cosine distance.
        This approach avoids SQLAlchemy type processing issues with Vector objects.

        Args:
            query_vector: The query embedding vector as a plain Python list of floats.
            vector_column: Name of the vector column to search.
            limit: Maximum number of results to return.

        Returns:
            List of entities ordered by similarity (most similar first).

        Note:
            Requires VectorChord extension and vchordrq index on the embedding column.
            Example index: CREATE INDEX ON table USING vchordrq (embedding vector_cosine_ops);
        """
        results_with_scores = self.vector_search_with_scores(query_vector, vector_column, limit)
        return [entity for entity, _ in results_with_scores]

    def vector_search_with_scores(
        self,
        query_vector: list[float],
        vector_column: str = "embedding",
        limit: int = 10,
    ) -> list[tuple[T, float]]:
        """Perform vector similarity search using VectorChord's cosine distance.

        Uses raw SQL with VectorChord's <=> operator for cosine distance.
        This approach avoids SQLAlchemy type processing issues with Vector objects.

        Args:
            query_vector: The query embedding vector as a plain Python list of floats.
            vector_column: Name of the vector column to search.
            limit: Maximum number of results to return.

        Returns:
            List of tuples (entity, distance_score) ordered by similarity.
            Lower distance scores indicate higher similarity.
            The score is the cosine distance, which is calculated as (1 - cosine_similarity).
            0 means identical, 2 means opposite, and 1 means orthogonal.

        Note:
            Requires VectorChord extension and vchordrq index on the embedding column.
            Example index: CREATE INDEX ON table USING vchordrq (embedding vector_cosine_ops);
        """
        if not query_vector:
            return []

        vec_str = _vec_to_pg_literal(query_vector)
        table_name = self.model_cls.__tablename__  # ty: ignore[possibly-missing-attribute]
        sql = text(f"""
            SELECT id, {vector_column} <=> '{vec_str}'::vector AS distance
            FROM {table_name}
            WHERE {vector_column} IS NOT NULL
            ORDER BY distance
            LIMIT :limit
        """)  # noqa: S608

        results = self.session.execute(sql, {"limit": limit}).fetchall()

        # Batch fetch entities to avoid N+1 queries
        ids = [row[0] for row in results]
        scores = {row[0]: float(row[1]) for row in results}
        entities = self.get_by_ids(ids)

        # Preserve order from SQL results
        entity_map = {e.id: e for e in entities}  # ty: ignore[unresolved-attribute]
        return [(entity_map[id_], scores[id_]) for id_ in ids if id_ in entity_map]

    def set_multi_vector_embedding(
        self,
        entity_id: int,
        embeddings: list[list[float]],
        vector_column: str = "embeddings",
        id_column: str = "id",
    ) -> bool:
        """Set multi-vector embedding for an entity using raw SQL.

        This method bypasses SQLAlchemy's type processing to properly
        format vector arrays for VectorChord compatibility.

        Args:
            entity_id: The entity's primary key.
            embeddings: List of embedding vectors (list of list of floats).
            vector_column: Name of the multi-vector column (default: "embeddings").
            id_column: Name of the primary key column (default: "id").

        Returns:
            True if update was successful, False otherwise.
        """
        if not embeddings:
            return False

        array_literal = _vecs_to_pg_array(embeddings)
        table_name = self.model_cls.__tablename__  # ty: ignore[possibly-missing-attribute]
        sql = text(f"UPDATE {table_name} SET {vector_column} = {array_literal} WHERE {id_column} = :entity_id")  # noqa: S608

        result = cast(CursorResult[Any], self.session.execute(sql, {"entity_id": entity_id}))
        return result.rowcount > 0

    def set_multi_vector_embeddings_batch(
        self,
        entity_ids: list[int],
        embeddings_list: list[list[list[float]]],
        vector_column: str = "embeddings",
        id_column: str = "id",
    ) -> int:
        """Batch set multi-vector embeddings for multiple entities.

        Args:
            entity_ids: List of entity primary keys.
            embeddings_list: List of multi-vector embeddings (one per entity).
            vector_column: Name of the multi-vector column (default: "embeddings").
            id_column: Name of the primary key column (default: "id").

        Returns:
            Number of entities successfully updated.
        """
        if len(entity_ids) != len(embeddings_list):
            raise LengthMismatchError("entity_ids", "embeddings_list")

        total_updated = 0
        for entity_id, embeddings in zip(entity_ids, embeddings_list, strict=True):
            if self.set_multi_vector_embedding(entity_id, embeddings, vector_column, id_column):
                total_updated += 1

        return total_updated

    def maxsim_search(
        self,
        query_vectors: list[list[float]],
        vector_column: str = "embeddings",
        limit: int = 10,
    ) -> list[tuple[T, float]]:
        """Perform MaxSim search using VectorChord's @# operator.

        MaxSim computes late interaction similarity: for each query vector,
        find the closest document vector, compute dot product, and sum results.

        Args:
            query_vectors: List of query embedding vectors (multi-vector query).
            vector_column: Name of the multi-vector column to search.
            limit: Maximum number of results to return.

        Returns:
            List of tuples (entity, distance_score) ordered by similarity.
            Lower distance scores indicate higher similarity.
            The distance score calculated by (1 - maxsim_score), thus the range is [-infinity, 0].
            You might normalize this score by dividing by the number of query vectors.

        Note:
            Requires VectorChord extension and vchordrq index with vector_maxsim_ops.
            Example index: CREATE INDEX ON table USING vchordrq (embeddings vector_maxsim_ops);
        """
        if not query_vectors:
            return []

        query_array = _vecs_to_pg_array(query_vectors)
        table_name = self.model_cls.__tablename__  # ty: ignore[possibly-missing-attribute]
        sql = text(f"""
            SELECT id, {vector_column} @# {query_array} AS distance
            FROM {table_name}
            WHERE {vector_column} IS NOT NULL
            ORDER BY distance
            LIMIT :limit
        """)  # noqa: S608

        results = self.session.execute(sql, {"limit": limit}).fetchall()

        # Batch fetch entities to avoid N+1 queries
        ids = [row[0] for row in results]
        scores = {row[0]: float(row[1]) for row in results}
        entities = self.get_by_ids(ids)

        # Preserve order from SQL results
        entity_map = {e.id: e for e in entities}  # ty: ignore[unresolved-attribute]
        return [(entity_map[id_], scores[id_]) for id_ in ids if id_ in entity_map]

    def maxsim_search_with_ids(
        self,
        query_vectors: list[list[float]],
        vector_column: str = "embeddings",
        id_column: str = "id",
        limit: int = 10,
    ) -> list[tuple[int, float]]:
        """Perform MaxSim search and return only IDs with scores.

        This is more efficient when you only need IDs and scores.

        Args:
            query_vectors: List of query embedding vectors (multi-vector query).
            vector_column: Name of the multi-vector column to search.
            id_column: Name of the primary key column.
            limit: Maximum number of results to return.

        Returns:
            List of tuples (entity_id, distance_score) ordered by similarity.
        """
        if not query_vectors:
            return []

        query_array = _vecs_to_pg_array(query_vectors)
        table_name = self.model_cls.__tablename__  # ty: ignore[possibly-missing-attribute]
        sql = text(f"""
            SELECT {id_column}, {vector_column} @# {query_array} AS distance
            FROM {table_name}
            WHERE {vector_column} IS NOT NULL
            ORDER BY distance
            LIMIT :limit
        """)  # noqa: S608

        results = self.session.execute(sql, {"limit": limit}).fetchall()
        return [(int(row[0]), float(row[1])) for row in results]


class BaseEmbeddingRepository(GenericRepository[T]):
    """Base repository with embedding-specific operations.
    This base class is made for schemas that have 'embedding' and 'embeddings' columns.
    """

    def _execute_with_offset_limit(self, stmt: Any, limit: int | None = None, offset: int | None = None) -> list[T]:
        """Helper to execute a statement with optional offset and limit."""
        if offset:
            stmt = stmt.offset(offset)
        if limit:
            stmt = stmt.limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def get_without_embeddings(self, limit: int | None = None, offset: int | None = None) -> list[T]:
        """Retrieve entities that do not have embeddings.

        Args:
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of entities without embeddings.
        """
        stmt = select(self.model_cls).where(self.model_cls.embedding.is_(None))  # ty: ignore[possibly-missing-attribute]
        return self._execute_with_offset_limit(stmt, limit, offset)

    def get_with_embeddings(self, limit: int | None = None, offset: int | None = None) -> list[T]:
        """Retrieve entities that have embeddings.

        Args:
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of entities with embeddings.
        """
        stmt = select(self.model_cls).where(self.model_cls.embedding.is_not(None))  # ty: ignore[possibly-missing-attribute]
        return self._execute_with_offset_limit(stmt, limit, offset)

    def get_without_multi_embeddings(self, limit: int | None = None, offset: int | None = None) -> list[T]:
        """Retrieve entities that do not have multi-vector embeddings.

        Args:
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of entities without multi-vector embeddings.
        """
        stmt = select(self.model_cls).where(self.model_cls.embeddings.is_(None))  # ty: ignore[possibly-missing-attribute]
        return self._execute_with_offset_limit(stmt, limit, offset)

    def get_with_multi_embeddings(self, limit: int | None = None, offset: int | None = None) -> list[T]:
        """Retrieve entities that have multi-vector embeddings.

        Args:
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of entities with multi-vector embeddings.
        """
        stmt = select(self.model_cls).where(self.model_cls.embeddings.is_not(None))  # ty: ignore[possibly-missing-attribute]
        return self._execute_with_offset_limit(stmt, limit, offset)

    # ==================== Count Methods ====================

    def count_without_embeddings(self) -> int:
        """Count entities without single-vector embeddings.

        Returns:
            Number of entities where embedding is NULL.
        """
        from sqlalchemy import func

        stmt = (
            select(func.count())
            .select_from(self.model_cls)
            .where(
                self.model_cls.embedding.is_(None)  # ty: ignore[possibly-missing-attribute]
            )
        )
        return self.session.execute(stmt).scalar() or 0

    def count_without_multi_embeddings(self) -> int:
        """Count entities without multi-vector embeddings.

        Returns:
            Number of entities where embeddings is NULL.
        """
        from sqlalchemy import func

        stmt = (
            select(func.count())
            .select_from(self.model_cls)
            .where(
                self.model_cls.embeddings.is_(None)  # ty: ignore[possibly-missing-attribute]
            )
        )
        return self.session.execute(stmt).scalar() or 0

    # ==================== BM25 Methods ====================

    def batch_update_bm25_tokens(self, tokenizer: str = "bert", batch_size: int = 1000) -> int:
        """Update bm25_tokens for entities in batches.

        Uses VectorChord-BM25's tokenize() function to generate bm25vector from text.
        Uses LIMIT + loop approach to avoid transaction/memory issues with large datasets.

        Args:
            tokenizer: Tokenizer to use (default: "bert").
                Available tokenizers (pg_tokenizer pre-built models):
                    - "bert": bert-base-uncased (Hugging Face) - Default
                    - "wiki_tocken": Wikitext-103 trained model
                    - "gemma2b": Google lightweight model (~100MB memory)
                    - "llmlingua2": Microsoft summarization model (~200MB memory)
                See: https://github.com/tensorchord/pg_tokenizer.rs/blob/main/docs/06-model.md
            batch_size: Number of entities to update per batch (default: 1000).

        Returns:
            Total number of entities updated.
        """
        table_name = self.model_cls.__tablename__  # ty: ignore[possibly-missing-attribute]
        total_updated = 0

        while True:
            sql = text(f"""
                UPDATE {table_name}
                SET bm25_tokens = tokenize(contents, :tokenizer)::bm25vector
                WHERE id IN (
                    SELECT id FROM {table_name}
                    WHERE bm25_tokens IS NULL
                      AND contents IS NOT NULL
                      AND contents != ''
                    LIMIT :batch_size
                )
            """)  # noqa: S608

            result = self.session.execute(sql, {"tokenizer": tokenizer, "batch_size": batch_size})
            updated = getattr(result, "rowcount", 0) or 0

            if updated == 0:
                break

            self.session.commit()
            total_updated += updated
            logger.info(f"Updated {total_updated} {table_name} with BM25 tokens")

        return total_updated

    def get_without_bm25_tokens(self, limit: int | None = None) -> list[T]:
        """Retrieve entities that don't have bm25_tokens.

        Args:
            limit: Maximum number of results to return.

        Returns:
            List of entities without bm25_tokens.
        """
        table_name = self.model_cls.__tablename__  # ty: ignore[possibly-missing-attribute]
        limit_clause = f"LIMIT {limit}" if limit else ""

        sql = text(f"""
            SELECT id FROM {table_name}
            WHERE bm25_tokens IS NULL
              AND contents IS NOT NULL
              AND contents != ''
            {limit_clause}
        """)  # noqa: S608

        result = self.session.execute(sql)
        ids = [row[0] for row in result.fetchall()]
        return self.get_by_ids(ids)

    def count_with_bm25_tokens(self) -> int:
        """Count entities that have bm25_tokens.

        Returns:
            Number of entities with bm25_tokens populated.
        """
        table_name = self.model_cls.__tablename__  # ty: ignore[possibly-missing-attribute]
        sql = text(f"SELECT COUNT(*) FROM {table_name} WHERE bm25_tokens IS NOT NULL")  # noqa: S608
        result = self.session.execute(sql)
        return result.scalar() or 0


def create_repository(session: Session, model_cls: type[T]) -> GenericRepository[T]:
    """Factory function to create a repository instance.

    Args:
        session: SQLAlchemy session.
        model_cls: The model class for the repository.

    Returns:
        A new GenericRepository instance.

    Example:
        >>> session = SessionFactory()
        >>> user_repo = create_repository(session, User)
        >>> user = user_repo.get_by_id(1)
    """
    return GenericRepository(session, model_cls)


@contextmanager
def repository_context(session_factory, model_cls: type[T]):
    """Context manager for quick repository operations.

    Combines UnitOfWork and Repository creation for simple use cases
    where you need to perform operations on a single model type.

    Args:
        session_factory: SQLAlchemy sessionmaker.
        model_cls: The model class for the repository.

    Yields:
        A tuple of (repository, unit_of_work) for operations.

    Example:
        >>> with repository_context(SessionFactory, User) as (repo, uow):
        ...     user = repo.get_by_id(1)
        ...     user.name = "New Name"
        ...     uow.commit()
    """
    with UnitOfWork(session_factory) as uow:
        if uow.session is None:
            raise NoSessionError
        repo = GenericRepository(uow.session, model_cls)
        yield repo, uow
