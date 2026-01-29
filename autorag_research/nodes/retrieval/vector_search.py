"""
Vector search retrieval module supporting single-vector and multi-vector embeddings.

This module implements vector search retrieval using PostgreSQL's VectorChord extension,
supporting both single-vector (cosine similarity) and multi-vector (MaxSim late interaction)
search modes, with configurable targets for text chunks, image chunks, or both.
"""

from enum import Enum
from typing import Any

from llama_index.core.base.embeddings.base import BaseEmbedding
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.embeddings.base import MultiVectorBaseEmbedding
from autorag_research.injection import with_embedding
from autorag_research.nodes import BaseModule
from autorag_research.orm.repository.chunk import ChunkRepository
from autorag_research.orm.repository.image_chunk import ImageChunkRepository

EMBEDDING_MODEL_TYPES = BaseEmbedding | MultiVectorBaseEmbedding


class RetrievalTarget(str, Enum):
    """Target for vector search retrieval."""

    TEXT_ONLY = "text"
    IMAGE_ONLY = "image"
    BOTH = "both"


class VectorSearchModule(BaseModule):
    """
    Vector search retrieval module supporting single-vector and multi-vector embeddings.

    This module performs vector-based retrieval using PostgreSQL's VectorChord extension.
    It automatically detects the embedding model type and uses the appropriate search method:
    - Single-vector (BaseEmbedding): Uses cosine distance via vector_search_with_scores()
    - Multi-vector (MultiVectorBaseEmbedding): Uses MaxSim via maxsim_search()

    Attributes:
        session_factory: SQLAlchemy sessionmaker for database connections.
        embedding_model: The embedding model instance or config name string.
        target: Which chunk types to search (TEXT_ONLY, IMAGE_ONLY, or BOTH).
        distance_threshold: Optional maximum distance threshold for filtering results.
        schema: Schema namespace from create_schema().

    Example:
        ```python
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        engine = create_engine("postgresql://user:pass@localhost/dbname")
        session_factory = sessionmaker(bind=engine)

        module = VectorSearchModule(
            session_factory=session_factory,
            embedding_model="openai-large",
            target=RetrievalTarget.TEXT_ONLY,
        )
        results = module.run(["What is machine learning?"], top_k=10)
        ```
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        embedding_model: str | EMBEDDING_MODEL_TYPES,
        target: RetrievalTarget = RetrievalTarget.TEXT_ONLY,
        distance_threshold: float | None = None,
        schema: Any | None = None,
    ):
        """
        Initialize vector search module.

        Args:
            session_factory: SQLAlchemy sessionmaker for database connections.
            embedding_model: The embedding model instance or config name string.
                Can be a LlamaIndex BaseEmbedding, MultiVectorBaseEmbedding, or a config name.
            target: Which chunk types to search (default: TEXT_ONLY).
            distance_threshold: Optional maximum distance threshold for filtering results.
            schema: Schema namespace from create_schema(). If None, uses default schema.
        """
        self.session_factory = session_factory
        self.embedding_model = embedding_model
        self.target = target
        self.distance_threshold = distance_threshold
        self._schema = schema

    def _get_chunk_model(self) -> type:
        """Get the Chunk model class from schema or default."""
        if self._schema is not None:
            return self._schema.Chunk
        from autorag_research.orm.schema import Chunk

        return Chunk

    def _get_image_chunk_model(self) -> type:
        """Get the ImageChunk model class from schema or default."""
        if self._schema is not None:
            return self._schema.ImageChunk
        from autorag_research.orm.schema import ImageChunk

        return ImageChunk

    def _is_multi_vector_model(self, model: EMBEDDING_MODEL_TYPES) -> bool:
        """Check if embedding model is multi-vector type.

        Args:
            model: The embedding model to check.

        Returns:
            True if the model is a MultiVectorBaseEmbedding, False otherwise.
        """
        return isinstance(model, MultiVectorBaseEmbedding)

    @with_embedding(param_name="embedding_model")
    def run(
        self,
        queries: list[str],
        top_k: int = 10,
        embedding_model: EMBEDDING_MODEL_TYPES | None = None,
    ) -> list[list[dict]]:
        """
        Execute vector search for given queries.

        Args:
            queries: List of query strings for batch processing.
            top_k: Number of top documents to retrieve per query.
            embedding_model: The embedding model instance (injected by @with_embedding if string was passed).

        Returns:
            List of search results for each query. Each query returns a list of top_k results.

            Each result dictionary contains:
            - doc_id: Chunk or ImageChunk ID (int)
            - score: Similarity score (higher = more relevant)
            - content: Chunk text content (None for image chunks)
            - chunk_type: "text" or "image"
        """
        # embedding_model is injected by @with_embedding if string was passed
        model = embedding_model or self.embedding_model
        if isinstance(model, str):
            # This shouldn't happen after @with_embedding, but handle it defensively
            raise TypeError(f"embedding_model must be resolved to an instance, got string: {model}")  # noqa: TRY003

        all_results = []
        with self.session_factory() as session:
            for query in queries:
                # Use isinstance directly for proper type narrowing
                if isinstance(model, MultiVectorBaseEmbedding):
                    query_embedding = model.get_query_embedding(query)
                    results = self._search_multi_vector(session, query_embedding, top_k)
                else:
                    # model is BaseEmbedding here
                    query_embedding = model.get_text_embedding(query)
                    results = self._search_single_vector(session, query_embedding, top_k)
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
        results = []

        if self.target in (RetrievalTarget.TEXT_ONLY, RetrievalTarget.BOTH):
            chunk_repo = ChunkRepository(session, self._get_chunk_model())
            chunk_results = chunk_repo.vector_search_with_scores(
                query_vector=query_embedding,
                limit=top_k if self.target == RetrievalTarget.TEXT_ONLY else top_k * 2,
                distance_threshold=self.distance_threshold,
            )
            for chunk, distance in chunk_results:
                results.append({
                    "doc_id": chunk.id,
                    "score": 1 - distance,  # Convert distance to similarity
                    "content": chunk.contents,
                    "chunk_type": "text",
                })

        if self.target in (RetrievalTarget.IMAGE_ONLY, RetrievalTarget.BOTH):
            image_repo = ImageChunkRepository(session, self._get_image_chunk_model())
            image_results = image_repo.vector_search_with_scores(
                query_vector=query_embedding,
                limit=top_k if self.target == RetrievalTarget.IMAGE_ONLY else top_k * 2,
                distance_threshold=self.distance_threshold,
            )
            for image_chunk, distance in image_results:
                results.append({
                    "doc_id": image_chunk.id,
                    "score": 1 - distance,
                    "content": None,  # Binary image data not included
                    "chunk_type": "image",
                })

        # Sort by score descending and return top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _search_multi_vector(self, session: Session, query_embeddings: list[list[float]], top_k: int) -> list[dict]:
        """Search using multi-vector MaxSim.

        Args:
            session: Database session.
            query_embeddings: Multi-vector query embeddings (list of vectors).
            top_k: Number of results to return.

        Returns:
            List of result dictionaries sorted by score descending.
        """
        results = []

        if self.target in (RetrievalTarget.TEXT_ONLY, RetrievalTarget.BOTH):
            chunk_repo = ChunkRepository(session, self._get_chunk_model())
            chunk_results = chunk_repo.maxsim_search(
                query_vectors=query_embeddings,
                vector_column="embeddings",
                limit=top_k if self.target == RetrievalTarget.TEXT_ONLY else top_k * 2,
            )
            for chunk, distance in chunk_results:
                results.append({
                    "doc_id": chunk.id,
                    "score": -distance,  # MaxSim: lower distance = higher similarity
                    "content": chunk.contents,
                    "chunk_type": "text",
                })

        if self.target in (RetrievalTarget.IMAGE_ONLY, RetrievalTarget.BOTH):
            image_repo = ImageChunkRepository(session, self._get_image_chunk_model())
            image_results = image_repo.maxsim_search(
                query_vectors=query_embeddings,
                vector_column="embeddings",
                limit=top_k if self.target == RetrievalTarget.IMAGE_ONLY else top_k * 2,
            )
            for image_chunk, distance in image_results:
                results.append({
                    "doc_id": image_chunk.id,
                    "score": -distance,
                    "content": None,
                    "chunk_type": "image",
                })

        # Sort by score descending and return top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
