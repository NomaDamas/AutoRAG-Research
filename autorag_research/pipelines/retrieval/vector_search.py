"""Vector Search Retrieval Pipeline for AutoRAG-Research.

This pipeline provides vector-based retrieval supporting both single-vector
(cosine similarity) and multi-vector (MaxSim late interaction) search modes
for text chunks.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.embeddings import Embeddings
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.config import BaseRetrievalPipelineConfig
from autorag_research.exceptions import EmbeddingError
from autorag_research.pipelines.retrieval.base import BaseRetrievalPipeline


@dataclass(kw_only=True)
class VectorSearchPipelineConfig(BaseRetrievalPipelineConfig):
    """Configuration for vector search retrieval pipeline.

    Attributes:
        name: Unique name for this pipeline instance.
        search_mode: Which embedding field to use ("single" or "multi").
        embedding_model: Optional embedding model for text-based retrieval.
        top_k: Number of results to retrieve per query.
        batch_size: Number of queries to process in each batch.

    Example:
        ```python
        config = VectorSearchPipelineConfig(
            name="vector_search_baseline",
            search_mode="single",
            top_k=10,
        )
        ```
    """

    search_mode: Literal["single", "multi"] = field(default="single")
    """Which embedding field to use for search.

    - "single": Uses query.embedding (single vector) with cosine similarity
    - "multi": Uses query.embeddings (multi-vector) with MaxSim
    """

    embedding_model: Embeddings | str | None = field(default=None)
    """Optional embedding model for text-based retrieval (_retrieve_by_text).

    Required when using retrieve() with queries that don't exist in the database.
    """

    def get_pipeline_class(self) -> type["VectorSearchRetrievalPipeline"]:
        """Return the VectorSearchRetrievalPipeline class."""
        return VectorSearchRetrievalPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        """Return kwargs for VectorSearchRetrievalPipeline constructor."""
        return {"search_mode": self.search_mode, "embedding_model": self.embedding_model}

    def __setattr__(self, name: str, value: Any) -> None:
        """Custom setattr to enforce type checks on search_mode."""
        if name == "embedding_model" and isinstance(value, str):
            from autorag_research.injection import load_embedding_model

            load_embedding_model(value)
        super().__setattr__(name, value)


class VectorSearchRetrievalPipeline(BaseRetrievalPipeline):
    """Pipeline for running vector search retrieval.

    This pipeline uses RetrievalPipelineService.vector_search() for
    vector-based retrieval using PostgreSQL's VectorChord extension.

    Supports both single-vector (cosine similarity) and multi-vector (MaxSim)
    search modes for text chunks. For batch processing (run()), queries must
    have pre-computed embeddings. For single-query retrieval (retrieve()),
    an embedding_model can be provided to compute embeddings on-the-fly.

    Example:
        ```python
        from autorag_research.orm.connection import DBConnection
        from autorag_research.pipelines.retrieval.vector_search import (
            VectorSearchRetrievalPipeline,
        )

        db = DBConnection.from_config()
        session_factory = db.get_session_factory()

        # Single-vector text search (default)
        pipeline = VectorSearchRetrievalPipeline(
            session_factory=session_factory,
            name="single_vector_search",
            search_mode="single",
        )

        # Run pipeline on all queries in DB
        results = pipeline.run(top_k=10)

        # Or with embedding model for ad-hoc queries
        from langchain_openai import OpenAIEmbeddings
        pipeline = VectorSearchRetrievalPipeline(
            session_factory=session_factory,
            name="vector_search_with_embed",
            embedding_model=OpenAIEmbeddings(),
        )
        results = await pipeline.retrieve("What is machine learning?", top_k=10)
        ```
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        search_mode: Literal["single", "multi"] = "single",
        embedding_model: Embeddings | None = None,
        schema: Any | None = None,
    ):
        """Initialize vector search retrieval pipeline.

        Args:
            session_factory: SQLAlchemy sessionmaker for database connections.
            name: Name for this pipeline.
            search_mode: Which embedding field to use for search.
                - "single": Uses query.embedding (single vector) with cosine similarity
                - "multi": Uses query.embeddings (multi-vector) with MaxSim
            embedding_model: Optional LangChain Embeddings instance for text-based retrieval.
                Required when using retrieve() with queries that don't exist in the database.
            schema: Schema namespace from create_schema(). If None, uses default schema.
        """
        # Store parameters BEFORE calling super().__init__
        # because _get_pipeline_config() is called in super().__init__
        self.search_mode = search_mode
        self._embedding_model = embedding_model

        super().__init__(session_factory, name, schema)

    def _get_pipeline_config(self) -> dict[str, Any]:
        """Return vector search pipeline configuration.

        Returns:
            Dictionary containing pipeline configuration for storage.
        """
        return {
            "type": "vector_search",
            "search_mode": self.search_mode,
        }

    async def _retrieve_by_id(self, query_id: int | str, top_k: int) -> list[dict[str, Any]]:
        """Vector search using query ID (uses stored query embedding).

        Args:
            query_id: The query ID to retrieve for.
            top_k: Number of top documents to retrieve.

        Returns:
            List of result dicts with doc_id, score, and content.
        """
        # Sync DB call (fast) - no need for true async
        results = self._service.vector_search([query_id], top_k, search_mode=self.search_mode)
        return results[0] if results else []

    async def _retrieve_by_text(self, query_text: str, top_k: int) -> list[dict[str, Any]]:
        """Vector search using raw text (computes embedding on-the-fly).

        Args:
            query_text: The query text to retrieve for.
            top_k: Number of top documents to retrieve.

        Returns:
            List of result dicts with doc_id, score, and content.

        Raises:
            ValueError: If embedding_model is not provided.
        """
        if self._embedding_model is None:
            raise EmbeddingError

        # Compute embedding on-the-fly
        query_embedding = await self._embedding_model.aembed_query(query_text)

        # Search directly with embedding vector
        return self._service.vector_search_by_embedding(query_embedding, top_k)


__all__ = ["VectorSearchPipelineConfig", "VectorSearchRetrievalPipeline"]
