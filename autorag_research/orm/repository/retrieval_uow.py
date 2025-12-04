"""Retrieval Unit of Work for managing retrieval pipeline transactions.

Provides atomic transaction management for retrieval operations including:
- Query fetching
- Chunk/ImageChunk retrieval for content
- Pipeline configuration
- Result storage (ChunkRetrievedResult)
"""

from typing import Protocol, runtime_checkable

from sqlalchemy.orm import Session, sessionmaker

from autorag_research.exceptions import SessionNotSetError
from autorag_research.orm.repository.base import UnitOfWork
from autorag_research.orm.repository.chunk import ChunkRepository
from autorag_research.orm.repository.chunk_retrieved_result import ChunkRetrievedResultRepository
from autorag_research.orm.repository.image_chunk import ImageChunkRepository
from autorag_research.orm.repository.metric import MetricRepository
from autorag_research.orm.repository.pipeline import PipelineRepository
from autorag_research.orm.repository.query import QueryRepository


@runtime_checkable
class RetrievalSchemaProtocol(Protocol):
    """Protocol defining the required schema classes for retrieval operations."""

    Query: type
    Chunk: type
    ImageChunk: type
    Pipeline: type
    Metric: type
    ChunkRetrievedResult: type


class RetrievalUnitOfWork(UnitOfWork):
    """Unit of Work for retrieval pipeline operations.

    Manages transactions across multiple repositories needed for retrieval:
    - Query: For fetching search queries
    - Chunk/ImageChunk: For retrieving original content
    - Pipeline: For configuration and tracking
    - Metric: For tracking evaluation metrics
    - ChunkRetrievedResult: For storing text retrieval results

    Note: ImageChunkRetrievedResult can be added later for multi-modal retrieval.
    """

    def __init__(self, session_factory: sessionmaker[Session], schema: RetrievalSchemaProtocol):
        """Initialize RetrievalUnitOfWork with session factory and schema.

        Args:
            session_factory: SQLAlchemy sessionmaker instance.
            schema: Schema namespace containing ORM models (must satisfy RetrievalSchemaProtocol).
        """
        super().__init__(session_factory)
        self.schema = schema
        self._query_repo: QueryRepository | None = None
        self._chunk_repo: ChunkRepository | None = None
        self._image_chunk_repo: ImageChunkRepository | None = None
        self._pipeline_repo: PipelineRepository | None = None
        self._metric_repo: MetricRepository | None = None
        self._chunk_result_repo: ChunkRetrievedResultRepository | None = None

    @property
    def queries(self) -> QueryRepository:
        """Get or create QueryRepository."""
        if self._query_repo is None:
            if self.session is None:
                raise SessionNotSetError
            self._query_repo = QueryRepository(self.session, self.schema.Query)
        return self._query_repo

    @property
    def chunks(self) -> ChunkRepository:
        """Get or create ChunkRepository."""
        if self._chunk_repo is None:
            if self.session is None:
                raise SessionNotSetError
            self._chunk_repo = ChunkRepository(self.session, self.schema.Chunk)
        return self._chunk_repo

    @property
    def image_chunks(self) -> ImageChunkRepository:
        """Get or create ImageChunkRepository."""
        if self._image_chunk_repo is None:
            if self.session is None:
                raise SessionNotSetError
            self._image_chunk_repo = ImageChunkRepository(self.session, self.schema.ImageChunk)
        return self._image_chunk_repo

    @property
    def pipelines(self) -> PipelineRepository:
        """Get or create PipelineRepository."""
        if self._pipeline_repo is None:
            if self.session is None:
                raise SessionNotSetError
            self._pipeline_repo = PipelineRepository(self.session, self.schema.Pipeline)
        return self._pipeline_repo

    @property
    def metrics(self) -> MetricRepository:
        """Get or create MetricRepository."""
        if self._metric_repo is None:
            if self.session is None:
                raise SessionNotSetError
            self._metric_repo = MetricRepository(self.session, self.schema.Metric)
        return self._metric_repo

    @property
    def chunk_results(self) -> ChunkRetrievedResultRepository:
        """Get or create ChunkRetrievedResultRepository."""
        if self._chunk_result_repo is None:
            if self.session is None:
                raise SessionNotSetError
            self._chunk_result_repo = ChunkRetrievedResultRepository(self.session, self.schema.ChunkRetrievedResult)
        return self._chunk_result_repo

    def __enter__(self) -> "RetrievalUnitOfWork":
        """Enter context and reset repository cache."""
        super().__enter__()
        # Reset repository cache for new session
        self._query_repo = None
        self._chunk_repo = None
        self._image_chunk_repo = None
        self._pipeline_repo = None
        self._metric_repo = None
        self._chunk_result_repo = None
        return self
