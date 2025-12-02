"""BM25 Pipeline Unit of Work for AutoRAG-Research.

Provides a specialized Unit of Work pattern for BM25 retrieval pipeline,
focusing on Query, Pipeline, Chunk, and ChunkRetrievedResult repositories.
"""

from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from autorag_research.exceptions import SessionNotSetError
from autorag_research.orm.repository.chunk import ChunkRepository
from autorag_research.orm.repository.chunk_retrieved_result import ChunkRetrievedResultRepository
from autorag_research.orm.repository.metric import MetricRepository
from autorag_research.orm.repository.pipeline import PipelineRepository
from autorag_research.orm.repository.query import QueryRepository


class BM25PipelineUnitOfWork:
    """Unit of Work for managing BM25 pipeline transactions.

    This UoW focuses on entities needed for BM25 retrieval:
    - Query: Input queries to retrieve
    - Pipeline: Pipeline configuration
    - Metric: Metric definition for the retrieval method
    - Chunk: Retrieved text chunks (for mapping doc_id to chunk_id)
    - ChunkRetrievedResult: Storage for retrieval results

    Provides lazy-initialized repositories for efficient resource usage.
    """

    def __init__(self, session_factory: sessionmaker[Session], schema: Any | None = None):
        """Initialize BM25 Pipeline Unit of Work with a session factory.

        Args:
            session_factory: SQLAlchemy sessionmaker instance.
            schema: Schema namespace from create_schema(). If None, uses default 768-dim schema.
        """
        self.session_factory = session_factory
        self._schema = schema
        self.session: Session | None = None
        self._query_repo: QueryRepository | None = None
        self._pipeline_repo: PipelineRepository | None = None
        self._metric_repo: MetricRepository | None = None
        self._chunk_repo: ChunkRepository | None = None
        self._chunk_retrieved_result_repo: ChunkRetrievedResultRepository | None = None

    def _get_schema_classes(self) -> dict[str, type]:
        """Get schema classes from schema namespace.

        Returns:
            Dictionary mapping class names to ORM classes.
        """
        if self._schema is not None:
            return {
                "Query": self._schema.Query,
                "Pipeline": self._schema.Pipeline,
                "Metric": self._schema.Metric,
                "Chunk": self._schema.Chunk,
                "ChunkRetrievedResult": self._schema.ChunkRetrievedResult,
            }
        # Use default schema
        from autorag_research.orm.schema import Chunk, ChunkRetrievedResult, Metric, Pipeline, Query

        return {
            "Query": Query,
            "Pipeline": Pipeline,
            "Metric": Metric,
            "Chunk": Chunk,
            "ChunkRetrievedResult": ChunkRetrievedResult,
        }

    def __enter__(self) -> "BM25PipelineUnitOfWork":
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
        self._pipeline_repo = None
        self._metric_repo = None
        self._chunk_repo = None
        self._chunk_retrieved_result_repo = None

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
            self._pipeline_repo = PipelineRepository(self.session)
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
            self._metric_repo = MetricRepository(self.session)
        return self._metric_repo

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
    def chunk_retrieved_results(self) -> ChunkRetrievedResultRepository:
        """Get the ChunkRetrievedResult repository.

        Returns:
            ChunkRetrievedResultRepository instance.

        Raises:
            SessionNotSetError: If session is not initialized.
        """
        if self.session is None:
            raise SessionNotSetError
        if self._chunk_retrieved_result_repo is None:
            classes = self._get_schema_classes()
            self._chunk_retrieved_result_repo = ChunkRetrievedResultRepository(
                self.session, classes["ChunkRetrievedResult"]
            )
        return self._chunk_retrieved_result_repo

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
