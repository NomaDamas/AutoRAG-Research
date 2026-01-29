"""Vector Search Retrieval Pipeline for AutoRAG-Research.

This pipeline provides vector-based retrieval supporting both single-vector
(cosine similarity) and multi-vector (MaxSim late interaction) search modes,
with configurable targets for text chunks, image chunks, or both.
"""

from dataclasses import dataclass
from typing import Any

from llama_index.core.base.embeddings.base import BaseEmbedding
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.config import BaseRetrievalPipelineConfig
from autorag_research.embeddings.base import MultiVectorBaseEmbedding
from autorag_research.nodes.retrieval.vector_search import RetrievalTarget
from autorag_research.pipelines.retrieval.base import BaseRetrievalPipeline

EMBEDDING_MODEL_TYPES = BaseEmbedding | MultiVectorBaseEmbedding


@dataclass(kw_only=True)
class VectorSearchPipelineConfig(BaseRetrievalPipelineConfig):
    """Configuration for vector search retrieval pipeline.

    Attributes:
        name: Unique name for this pipeline instance.
        embedding_model: The embedding model instance or config name string.
        target: Which chunk types to search (TEXT_ONLY, IMAGE_ONLY, or BOTH).
        distance_threshold: Optional maximum distance threshold for filtering results.
        top_k: Number of results to retrieve per query.
        batch_size: Number of queries to process in each batch.

    Example:
        ```python
        config = VectorSearchPipelineConfig(
            name="vector_search_baseline",
            embedding_model="openai-large",
            target=RetrievalTarget.TEXT_ONLY,
            top_k=10,
        )
        ```
    """

    embedding_model: str | EMBEDDING_MODEL_TYPES
    """The embedding model instance or config name string.

    Can be:
    - A string config name (e.g., "openai-large") that will be loaded via Hydra
    - A LlamaIndex BaseEmbedding instance for single-vector search
    - A MultiVectorBaseEmbedding instance for MaxSim late interaction search
    """

    target: RetrievalTarget = RetrievalTarget.TEXT_ONLY
    """Which chunk types to search.

    Options:
    - TEXT_ONLY: Search only text chunks (Chunk table)
    - IMAGE_ONLY: Search only image chunks (ImageChunk table)
    - BOTH: Search both, merge results by score
    """

    distance_threshold: float | None = None
    """Optional maximum distance threshold for filtering results.

    Only applies to single-vector search. Results with distance greater
    than this threshold will be filtered out.
    """

    def get_pipeline_class(self) -> type["VectorSearchRetrievalPipeline"]:
        """Return the VectorSearchRetrievalPipeline class."""
        return VectorSearchRetrievalPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        """Return kwargs for VectorSearchRetrievalPipeline constructor."""
        return {
            "embedding_model": self.embedding_model,
            "target": self.target,
            "distance_threshold": self.distance_threshold,
        }


class VectorSearchRetrievalPipeline(BaseRetrievalPipeline):
    """Pipeline for running vector search retrieval.

    This pipeline wraps RetrievalPipelineService with VectorSearchModule,
    providing a convenient interface for vector-based retrieval using
    PostgreSQL's VectorChord extension.

    Supports both single-vector (cosine similarity) and multi-vector (MaxSim)
    search modes, with configurable targets for text chunks, image chunks, or both.

    Example:
        ```python
        from autorag_research.orm.connection import DBConnection
        from autorag_research.pipelines.retrieval.vector_search import (
            VectorSearchRetrievalPipeline,
            RetrievalTarget,
        )

        db = DBConnection.from_config()
        session_factory = db.get_session_factory()

        # Single-vector text search
        pipeline = VectorSearchRetrievalPipeline(
            session_factory=session_factory,
            name="openai_vector_search",
            embedding_model="openai-large",
            target=RetrievalTarget.TEXT_ONLY,
        )

        # Run pipeline on all queries in DB
        results = pipeline.run(top_k=10)

        # Or retrieve for a single query
        chunks = pipeline.retrieve("What is machine learning?", top_k=10)
        ```
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        embedding_model: str | EMBEDDING_MODEL_TYPES,
        target: RetrievalTarget = RetrievalTarget.TEXT_ONLY,
        distance_threshold: float | None = None,
        schema: Any | None = None,
    ):
        """Initialize vector search retrieval pipeline.

        Args:
            session_factory: SQLAlchemy sessionmaker for database connections.
            name: Name for this pipeline.
            embedding_model: The embedding model instance or config name string.
                Can be a LlamaIndex BaseEmbedding, MultiVectorBaseEmbedding, or a config name.
            target: Which chunk types to search (default: TEXT_ONLY).
            distance_threshold: Optional maximum distance threshold for filtering results.
            schema: Schema namespace from create_schema(). If None, uses default schema.
        """
        # Store parameters BEFORE calling super().__init__
        # because _get_pipeline_config() is called in super().__init__
        self.embedding_model = embedding_model
        self.target = target
        self.distance_threshold = distance_threshold

        super().__init__(session_factory, name, schema)

    def _get_pipeline_config(self) -> dict[str, Any]:
        """Return vector search pipeline configuration.

        Returns:
            Dictionary containing pipeline configuration for storage.
            The embedding_model is serialized to a string for DB storage.
        """
        # Serialize embedding_model to string for DB storage
        if isinstance(self.embedding_model, str):
            model_name = self.embedding_model
        else:
            model_name = getattr(self.embedding_model, "model_name", str(type(self.embedding_model).__name__))

        return {
            "type": "vector_search",
            "embedding_model": model_name,
            "target": self.target.value,
            "distance_threshold": self.distance_threshold,
        }

    def _get_retrieval_func(self) -> Any:
        """Return vector search retrieval function.

        Returns:
            The VectorSearchModule.run method configured with pipeline parameters.
        """
        from autorag_research.nodes.retrieval.vector_search import VectorSearchModule

        module = VectorSearchModule(
            session_factory=self.session_factory,
            embedding_model=self.embedding_model,
            target=self.target,
            distance_threshold=self.distance_threshold,
            schema=self._schema,
        )
        return module.run


# Re-export RetrievalTarget for convenience
__all__ = ["RetrievalTarget", "VectorSearchPipelineConfig", "VectorSearchRetrievalPipeline"]
