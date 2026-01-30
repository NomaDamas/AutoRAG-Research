"""Vector Search Retrieval Pipeline for AutoRAG-Research.

This pipeline provides vector-based retrieval supporting both single-vector
(cosine similarity) and multi-vector (MaxSim late interaction) search modes
for text chunks.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

from sqlalchemy.orm import Session, sessionmaker

from autorag_research.config import BaseRetrievalPipelineConfig
from autorag_research.pipelines.retrieval.base import BaseRetrievalPipeline


@dataclass(kw_only=True)
class VectorSearchPipelineConfig(BaseRetrievalPipelineConfig):
    """Configuration for vector search retrieval pipeline.

    Attributes:
        name: Unique name for this pipeline instance.
        search_mode: Which embedding field to use ("single" or "multi").
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

    def get_pipeline_class(self) -> type["VectorSearchRetrievalPipeline"]:
        """Return the VectorSearchRetrievalPipeline class."""
        return VectorSearchRetrievalPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        """Return kwargs for VectorSearchRetrievalPipeline constructor."""
        return {"search_mode": self.search_mode}


class VectorSearchRetrievalPipeline(BaseRetrievalPipeline):
    """Pipeline for running vector search retrieval.

    This pipeline wraps RetrievalPipelineService with VectorSearchModule,
    providing a convenient interface for vector-based retrieval using
    PostgreSQL's VectorChord extension.

    Supports both single-vector (cosine similarity) and multi-vector (MaxSim)
    search modes for text chunks. Queries must have pre-computed embeddings
    (via DataIngestor.embed_all()).

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
        ```
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        search_mode: Literal["single", "multi"] = "single",
        schema: Any | None = None,
    ):
        """Initialize vector search retrieval pipeline.

        Args:
            session_factory: SQLAlchemy sessionmaker for database connections.
            name: Name for this pipeline.
            search_mode: Which embedding field to use for search.
                - "single": Uses query.embedding (single vector) with cosine similarity
                - "multi": Uses query.embeddings (multi-vector) with MaxSim
            schema: Schema namespace from create_schema(). If None, uses default schema.
        """
        # Store parameters BEFORE calling super().__init__
        # because _get_pipeline_config() is called in super().__init__
        self.search_mode = search_mode

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

    def _get_retrieval_func(self) -> Any:
        """Return vector search retrieval function.

        Returns:
            A callable that invokes the service's vector_search method.
        """
        return lambda query_ids, top_k: self._service.vector_search(query_ids, top_k, search_mode=self.search_mode)


__all__ = ["VectorSearchPipelineConfig", "VectorSearchRetrievalPipeline"]
