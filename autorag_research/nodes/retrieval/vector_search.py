"""
Vector search retrieval module supporting single-vector and multi-vector embeddings.

This module implements vector search retrieval using PostgreSQL's VectorChord extension,
supporting both single-vector (cosine similarity) and multi-vector (MaxSim late interaction)
search modes for text chunks.
"""

from typing import Any, Literal

from sqlalchemy.orm import Session, sessionmaker

from autorag_research.nodes import BaseModule
from autorag_research.orm.repository.chunk import ChunkRepository
from autorag_research.orm.repository.query import QueryRepository


class VectorSearchModule(BaseModule):
    """
    Vector search retrieval module supporting single-vector and multi-vector embeddings.

    This module performs vector-based retrieval using PostgreSQL's VectorChord extension.
    Queries must have pre-computed embeddings (via DataIngestor.embed_all()).
    The search mode determines which embedding field to use:
    - single: Uses query.embedding (single vector) with cosine distance
    - multi: Uses query.embeddings (multiple vectors) with MaxSim

    Attributes:
        session_factory: SQLAlchemy sessionmaker for database connections.
        search_mode: Which embedding field to use ("single" or "multi").
        schema: Schema namespace from create_schema().

    Example:
        ```python
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        engine = create_engine("postgresql://user:pass@localhost/dbname")
        session_factory = sessionmaker(bind=engine)

        module = VectorSearchModule(
            session_factory=session_factory,
            search_mode="single",
        )
        results = module.run([1, 2, 3], top_k=10)  # Query IDs
        ```
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        search_mode: Literal["single", "multi"] = "single",
        schema: Any | None = None,
    ):
        """
        Initialize vector search module.

        Args:
            session_factory: SQLAlchemy sessionmaker for database connections.
            search_mode: Which embedding field to use for search.
                - "single": Uses query.embedding (single vector) with cosine similarity
                - "multi": Uses query.embeddings (multi-vector) with MaxSim
            schema: Schema namespace from create_schema(). If None, uses default schema.
        """
        self.session_factory = session_factory
        self.search_mode = search_mode
        self._schema = schema

    def _get_chunk_model(self) -> type:
        """Get the Chunk model class from schema or default."""
        if self._schema is not None:
            return self._schema.Chunk
        from autorag_research.orm.schema import Chunk

        return Chunk

    def _get_query_model(self) -> type:
        """Get the Query model class from schema or default."""
        if self._schema is not None:
            return self._schema.Query
        from autorag_research.orm.schema import Query

        return Query

    def run(
        self,
        query_ids: list[int | str],
        top_k: int = 10,
    ) -> list[list[dict]]:
        """
        Execute vector search for given query IDs.

        Args:
            query_ids: List of query IDs to search. Queries must have pre-computed embeddings.
            top_k: Number of top documents to retrieve per query.

        Returns:
            List of search results for each query. Each query returns a list of top_k results.

            Each result dictionary contains:
            - doc_id: Chunk ID (int)
            - score: Similarity score (higher = more relevant)
            - content: Chunk text content

        Raises:
            ValueError: If a query is not found or has no embedding for the search mode.
        """
        all_results = []
        with self.session_factory() as session:
            query_repo = QueryRepository(session, self._get_query_model())
            for query_id in query_ids:
                query = query_repo.get_by_id(query_id)
                if query is None:
                    raise ValueError(f"Query {query_id} not found")  # noqa: TRY003

                if self.search_mode == "multi":
                    if query.embeddings is None:
                        raise ValueError(f"Query {query_id} has no multi-vector embeddings")  # noqa: TRY003
                    results = self._search_multi_vector(session, query.embeddings, top_k)
                else:
                    if query.embedding is None:
                        raise ValueError(f"Query {query_id} has no embedding")  # noqa: TRY003
                    results = self._search_single_vector(session, list(query.embedding), top_k)
                all_results.append(results)
        return all_results

    def _search_single_vector(self, session: Session, query_embedding: list[float], top_k: int) -> list[dict]:
        """Search using single-vector cosine similarity.

        Args:
            session: Database session.
            query_embedding: Single query embedding vector.
            top_k: Number of results to return.

        Returns:
            List of result dictionaries sorted by score descending.
        """
        chunk_repo = ChunkRepository(session, self._get_chunk_model())
        chunk_results = chunk_repo.vector_search_with_scores(
            query_vector=query_embedding,
            limit=top_k,
        )
        results = []
        for chunk, distance in chunk_results:
            results.append({
                "doc_id": chunk.id,
                "score": 1 - distance,  # Convert distance to similarity
                "content": chunk.contents,
            })
        return results

    def _search_multi_vector(self, session: Session, query_embeddings: list[list[float]], top_k: int) -> list[dict]:
        """Search using multi-vector MaxSim.

        Args:
            session: Database session.
            query_embeddings: Multi-vector query embeddings (list of vectors).
            top_k: Number of results to return.

        Returns:
            List of result dictionaries sorted by score descending.
        """
        chunk_repo = ChunkRepository(session, self._get_chunk_model())
        chunk_results = chunk_repo.maxsim_search(
            query_vectors=query_embeddings,
            vector_column="embeddings",
            limit=top_k,
        )
        results = []
        for chunk, distance in chunk_results:
            results.append({
                "doc_id": chunk.id,
                "score": -distance,  # MaxSim: lower distance = higher similarity
                "content": chunk.contents,
            })
        return results
