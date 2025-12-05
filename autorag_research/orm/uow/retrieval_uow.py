"""Retrieval Unit of Work for managing retrieval pipeline transactions.

Provides atomic transaction management for retrieval operations including:
- Query fetching
- Chunk/ImageChunk retrieval for content
- Pipeline configuration
- Result storage (ChunkRetrievedResult)
"""

from typing import Any

from sqlalchemy.orm import sessionmaker

from autorag_research.orm.repository.chunk import ChunkRepository
from autorag_research.orm.repository.chunk_retrieved_result import ChunkRetrievedResultRepository
from autorag_research.orm.repository.image_chunk import ImageChunkRepository
from autorag_research.orm.repository.metric import MetricRepository
from autorag_research.orm.repository.pipeline import PipelineRepository
from autorag_research.orm.repository.query import QueryRepository
from autorag_research.orm.uow.base import BaseUnitOfWork


class RetrievalUnitOfWork(BaseUnitOfWork):
    """Unit of Work for retrieval pipeline operations.

    Manages transactions across multiple repositories needed for retrieval:
    - Query: For fetching search queries
    - Chunk/ImageChunk: For retrieving original content
    - Pipeline: For configuration and tracking
    - Metric: For tracking evaluation metrics
    - ChunkRetrievedResult: For storing text retrieval results

    Note: ImageChunkRetrievedResult can be added later for multi-modal retrieval.
    """

    def __init__(self, session_factory: sessionmaker, schema: Any | None = None):
        """Initialize RetrievalUnitOfWork with session factory and schema.

        Args:
            session_factory: SQLAlchemy sessionmaker instance.
            schema: Schema namespace from create_schema(). If None, uses default 768-dim schema.
        """
        super().__init__(session_factory, schema)

        # Lazy-initialized repositories
        self._query_repo: QueryRepository | None = None
        self._chunk_repo: ChunkRepository | None = None
        self._image_chunk_repo: ImageChunkRepository | None = None
        self._pipeline_repo: PipelineRepository | None = None
        self._metric_repo: MetricRepository | None = None
        self._chunk_result_repo: ChunkRetrievedResultRepository | None = None

    def _get_schema_classes(self) -> dict[str, type]:
        """Get all model classes from schema.

        Returns:
            Dictionary mapping class names to model classes.
        """
        if self._schema is not None:
            return {
                "Query": self._schema.Query,
                "Chunk": self._schema.Chunk,
                "ImageChunk": self._schema.ImageChunk,
                "Pipeline": self._schema.Pipeline,
                "Metric": self._schema.Metric,
                "ChunkRetrievedResult": self._schema.ChunkRetrievedResult,
            }

        from autorag_research.orm.schema import (
            Chunk,
            ChunkRetrievedResult,
            ImageChunk,
            Metric,
            Pipeline,
            Query,
        )

        return {
            "Query": Query,
            "Chunk": Chunk,
            "ImageChunk": ImageChunk,
            "Pipeline": Pipeline,
            "Metric": Metric,
            "ChunkRetrievedResult": ChunkRetrievedResult,
        }

    def _reset_repositories(self) -> None:
        """Reset all repository references to None."""
        self._query_repo = None
        self._chunk_repo = None
        self._image_chunk_repo = None
        self._pipeline_repo = None
        self._metric_repo = None
        self._chunk_result_repo = None

    @property
    def queries(self) -> QueryRepository:
        """Get the Query repository.

        Returns:
            QueryRepository instance.

        Raises:
            SessionNotSetError: If session is not initialized.
        """
        return self._get_repository(
            "_query_repo",
            QueryRepository,
            lambda: self._get_schema_classes()["Query"],
        )

    @property
    def chunks(self) -> ChunkRepository:
        """Get the Chunk repository.

        Returns:
            ChunkRepository instance.

        Raises:
            SessionNotSetError: If session is not initialized.
        """
        return self._get_repository(
            "_chunk_repo",
            ChunkRepository,
            lambda: self._get_schema_classes()["Chunk"],
        )

    @property
    def image_chunks(self) -> ImageChunkRepository:
        """Get the ImageChunk repository.

        Returns:
            ImageChunkRepository instance.

        Raises:
            SessionNotSetError: If session is not initialized.
        """
        return self._get_repository(
            "_image_chunk_repo",
            ImageChunkRepository,
            lambda: self._get_schema_classes()["ImageChunk"],
        )

    @property
    def pipelines(self) -> PipelineRepository:
        """Get the Pipeline repository.

        Returns:
            PipelineRepository instance.

        Raises:
            SessionNotSetError: If session is not initialized.
        """
        return self._get_repository(
            "_pipeline_repo",
            PipelineRepository,
            lambda: self._get_schema_classes()["Pipeline"],
        )

    @property
    def metrics(self) -> MetricRepository:
        """Get the Metric repository.

        Returns:
            MetricRepository instance.

        Raises:
            SessionNotSetError: If session is not initialized.
        """
        return self._get_repository(
            "_metric_repo",
            MetricRepository,
            lambda: self._get_schema_classes()["Metric"],
        )

    @property
    def chunk_results(self) -> ChunkRetrievedResultRepository:
        """Get the ChunkRetrievedResult repository.

        Returns:
            ChunkRetrievedResultRepository instance.

        Raises:
            SessionNotSetError: If session is not initialized.
        """
        return self._get_repository(
            "_chunk_result_repo",
            ChunkRetrievedResultRepository,
            lambda: self._get_schema_classes()["ChunkRetrievedResult"],
        )
