"""
Vector search retrieval module supporting single-vector and multi-vector embeddings.

This module implements vector search retrieval using PostgreSQL's VectorChord extension,
supporting both single-vector (cosine similarity) and multi-vector (MaxSim late interaction)
search modes for text chunks.
"""

from typing import Any, Literal

from llama_index.core.embeddings import BaseEmbedding
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.embeddings.base import MultiVectorBaseEmbedding
from autorag_research.injection import with_embedding
from autorag_research.nodes import BaseModule, make_retrieval_result
from autorag_research.orm.repository.chunk import ChunkRepository
from autorag_research.orm.repository.query import QueryRepository
from autorag_research.util import to_list


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

    def search(
        self,
        query_embeddings: list[list[float]] | list[list[list[float]]],
        top_k: int = 10,
    ) -> list[list[dict]]:
        """
        Execute vector search with provided embeddings directly.

        Bypasses Query table lookup - use for HyDE, query expansion, etc.
        The search mode is automatically inferred from the embedding structure:
        - If each item is a 1D list (single embedding), uses single-vector search
        - If each item is a 2D list (multiple embeddings), uses multi-vector MaxSim search

        Args:
            query_embeddings: Embeddings to search with.
                - Single-vector mode: list[list[float]] where each inner list is an embedding
                - Multi-vector mode: list[list[list[float]]] where each query has multiple vectors
            top_k: Number of results per query.

        Returns:
            List of search results for each query. Each query returns a list of top_k results.

            Each result dictionary contains:
            - doc_id: Chunk ID (int)
            - score: Similarity score (higher = more relevant)
            - content: Chunk text content

        Example:
            ```python
            # Single-vector search (e.g., HyDE)
            hypothetical_embedding = embed_model.get_text_embedding("hypothetical doc")
            results = module.search([hypothetical_embedding], top_k=10)

            # Multi-vector search (e.g., ColBERT-style)
            token_embeddings = [[0.1, 0.2, ...], [0.3, 0.4, ...]]  # Multiple vectors per query
            results = module.search([token_embeddings], top_k=10)
            ```
        """
        if not query_embeddings:
            return []

        all_results = []
        with self.session_factory() as session:
            for embedding in query_embeddings:
                # Infer mode from embedding structure
                # If first element is a list of lists, it's multi-vector
                if embedding and isinstance(embedding[0], list):
                    # Multi-vector mode
                    results = self._search_multi_vector(session, to_list(embedding), top_k)
                else:
                    # Single-vector mode
                    results = self._search_single_vector(session, to_list(embedding), top_k)
                all_results.append(results)
        return all_results

    @with_embedding()
    def search_by_text(
        self,
        query_texts: list[str],
        embedding_model: BaseEmbedding | MultiVectorBaseEmbedding | str,
        top_k: int = 10,
    ) -> list[list[dict]]:
        """
        Embed texts at runtime and execute vector search.

        Convenience method that handles embedding generation internally.
        Always uses single-vector search mode.

        Args:
            query_texts: Texts to embed and search.
            embedding_model: LlamaIndex BaseEmbedding model instance or config name string.
                If string, loads model from config file (e.g., "openai-large").
            top_k: Number of results per query.

        Returns:
            List of search results for each query. Each query returns a list of top_k results.

            Each result dictionary contains:
            - doc_id: Chunk ID (int)
            - score: Similarity score (higher = more relevant)
            - content: Chunk text content

        Example:
            ```python
            # With model instance
            from llama_index.embeddings.openai import OpenAIEmbedding
            embed_model = OpenAIEmbedding()
            results = module.search_by_text(
                ["What is machine learning?"],
                embedding_model=embed_model,
                top_k=10
            )

            # With config name (loads from config/embedding/openai-large.yaml)
            results = module.search_by_text(
                ["What is machine learning?"],
                embedding_model="openai-large",
                top_k=10
            )
            ```
        """
        if not query_texts:
            return []

        # Type guard: decorator ensures embedding_model is BaseEmbedding at runtime
        if not isinstance(embedding_model, BaseEmbedding) or not isinstance(embedding_model, MultiVectorBaseEmbedding):
            raise TypeError(f"embedding_model must be BaseEmbedding or MultiVectorBaseEmbedding, got {type(embedding_model)}")  # noqa: TRY003

        # Generate embeddings using the provided model
        embeddings = embedding_model.get_text_embedding_batch(query_texts)
        return self.search(embeddings, top_k)

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
                    results = self._search_multi_vector(session, to_list(query.embeddings), top_k)
                else:
                    if query.embedding is None:
                        raise ValueError(f"Query {query_id} has no embedding")  # noqa: TRY003
                    results = self._search_single_vector(session, to_list(query.embedding), top_k)
                all_results.append(results)
        return all_results

    def _search_single_vector(self, session: Session, query_embedding: list[float], top_k: int) -> list[dict]:
        """Search using single-vector cosine similarity."""
        chunk_repo = ChunkRepository(session, self._get_chunk_model())
        chunk_results = chunk_repo.vector_search_with_scores(
            query_vector=query_embedding,
            limit=top_k,
        )
        return [make_retrieval_result(chunk, 1 - distance) for chunk, distance in chunk_results]

    def _search_multi_vector(self, session: Session, query_embeddings: list[list[float]], top_k: int) -> list[dict]:
        """Search using multi-vector MaxSim."""
        chunk_repo = ChunkRepository(session, self._get_chunk_model())
        chunk_results = chunk_repo.maxsim_search(
            query_vectors=query_embeddings,
            vector_column="embeddings",
            limit=top_k,
        )
        return [make_retrieval_result(chunk, -distance) for chunk, distance in chunk_results]
