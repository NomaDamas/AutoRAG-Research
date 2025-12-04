"""Retrieval Unit of Work for managing retrieval pipeline transactions.

Provides atomic transaction management for retrieval operations including:
- Query fetching
- Chunk/ImageChunk retrieval for content
- Pipeline configuration
- Result storage (ChunkRetrievedResult)
"""

from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from autorag_research.exceptions import SessionNotSetError
from autorag_research.orm.repository.chunk import ChunkRepository
from autorag_research.orm.repository.chunk_retrieved_result import ChunkRetrievedResultRepository
from autorag_research.orm.repository.image_chunk import ImageChunkRepository
from autorag_research.orm.repository.metric import MetricRepository
from autorag_research.orm.repository.pipeline import PipelineRepository
from autorag_research.orm.repository.query import QueryRepository


class RetrievalUnitOfWork:
    """Unit of Work for retrieval pipeline operations.

    Manages transactions across multiple repositories needed for retrieval:
    - Query: For fetching search queries
    - Chunk/ImageChunk: For retrieving original content
    - Pipeline: For configuration and tracking
    - Metric: For tracking evaluation metrics
    - ChunkRetrievedResult: For storing text retrieval results

    Note: ImageChunkRetrievedResult can be added later for multi-modal retrieval.
    """

    def __init__(self, session_factory: sessionmaker[Session], schema: Any | None = None):
        """Initialize RetrievalUnitOfWork with session factory and schema.

        Args:
            session_factory: SQLAlchemy sessionmaker instance.
            schema: Schema namespace from create_schema(). If None, uses default 768-dim schema.
        """
        self.session_factory = session_factory
        self._schema = schema
        self.session: Session | None = None

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
        # Use default schema
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

    def __enter__(self) -> "RetrievalUnitOfWork":
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
        # Reset repository references
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
        if self.session is None:
            raise SessionNotSetError
        if self._query_repo is None:
            classes = self._get_schema_classes()
            self._query_repo = QueryRepository(self.session, classes["Query"])
        return self._query_repo

    @property
    def chunks(self) -> ChunkRepository:
        """Get the Chunk repository.

        Returns:
            ChunkRepository instance.

        Raises:
            SessionNotSetError: If session is not initialized.
        """
        if self.session is None:
            raise SessionNotSetError
        if self._chunk_repo is None:
            classes = self._get_schema_classes()
            self._chunk_repo = ChunkRepository(self.session, classes["Chunk"])
        return self._chunk_repo

    @property
    def image_chunks(self) -> ImageChunkRepository:
        """Get the ImageChunk repository.

        Returns:
            ImageChunkRepository instance.

        Raises:
            SessionNotSetError: If session is not initialized.
        """
        if self.session is None:
            raise SessionNotSetError
        if self._image_chunk_repo is None:
            classes = self._get_schema_classes()
            self._image_chunk_repo = ImageChunkRepository(self.session, classes["ImageChunk"])
        return self._image_chunk_repo

    @property
    def pipelines(self) -> PipelineRepository:
        """Get the Pipeline repository.

        Returns:
            PipelineRepository instance.

        Raises:
            SessionNotSetError: If session is not initialized.
        """
        if self.session is None:
            raise SessionNotSetError
        if self._pipeline_repo is None:
            classes = self._get_schema_classes()
            self._pipeline_repo = PipelineRepository(self.session, classes["Pipeline"])
        return self._pipeline_repo

    @property
    def metrics(self) -> MetricRepository:
        """Get the Metric repository.

        Returns:
            MetricRepository instance.

        Raises:
            SessionNotSetError: If session is not initialized.
        """
        if self.session is None:
            raise SessionNotSetError
        if self._metric_repo is None:
            classes = self._get_schema_classes()
            self._metric_repo = MetricRepository(self.session, classes["Metric"])
        return self._metric_repo

    @property
    def chunk_results(self) -> ChunkRetrievedResultRepository:
        """Get the ChunkRetrievedResult repository.

        Returns:
            ChunkRetrievedResultRepository instance.

        Raises:
            SessionNotSetError: If session is not initialized.
        """
        if self.session is None:
            raise SessionNotSetError
        if self._chunk_result_repo is None:
            classes = self._get_schema_classes()
            self._chunk_result_repo = ChunkRetrievedResultRepository(self.session, classes["ChunkRetrievedResult"])
        return self._chunk_result_repo

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
