"""Generation Unit of Work for managing generation pipeline transactions.

Provides atomic transaction management for generation operations including:
- Query fetching
- Chunk content retrieval (for building context)
- Chunk retrieved results storage (for traceability)
- Executor results (generation outputs)
- Pipeline configuration
"""

from typing import Any

from sqlalchemy.orm import sessionmaker

from autorag_research.orm.repository.chunk import ChunkRepository
from autorag_research.orm.repository.chunk_retrieved_result import ChunkRetrievedResultRepository
from autorag_research.orm.repository.executor_result import ExecutorResultRepository
from autorag_research.orm.repository.pipeline import PipelineRepository
from autorag_research.orm.repository.query import QueryRepository
from autorag_research.orm.uow.base import BaseUnitOfWork


class GenerationUnitOfWork(BaseUnitOfWork):
    """Unit of Work for generation pipeline operations.

    Manages transactions across multiple repositories needed for generation:
    - Query: For fetching queries to process
    - Chunk: For getting chunk contents to build context
    - ChunkRetrievedResult: For storing retrieval results (traceability)
    - ExecutorResult: For storing generation outputs
    - Pipeline: For configuration and tracking

    Example:
        ```python
        with GenerationUnitOfWork(session_factory) as uow:
            # Fetch queries
            queries = uow.queries.get_all(limit=100)

            # Get chunk contents for context
            chunks = uow.chunks.get_by_ids([1, 2, 3])

            # Store retrieval results
            uow.chunk_results.bulk_insert([...])

            # Store generation results
            executor_result = ExecutorResult(
                query_id=query_id,
                pipeline_id=pipeline_id,
                generation_result="...",
                token_usage={"prompt_tokens": 50, "completion_tokens": 50, "total_tokens": 100},
            )
            uow.executor_results.add(executor_result)
            uow.commit()
        ```
    """

    def __init__(self, session_factory: sessionmaker, schema: Any | None = None):
        """Initialize GenerationUnitOfWork with session factory and schema.

        Args:
            session_factory: SQLAlchemy sessionmaker instance.
            schema: Schema namespace from create_schema(). If None, uses default 768-dim schema.
        """
        super().__init__(session_factory, schema)

        # Lazy-initialized repositories
        self._query_repo: QueryRepository | None = None
        self._chunk_repo: ChunkRepository | None = None
        self._chunk_result_repo: ChunkRetrievedResultRepository | None = None
        self._executor_result_repo: ExecutorResultRepository | None = None
        self._pipeline_repo: PipelineRepository | None = None

    def _get_schema_classes(self) -> dict[str, type]:
        """Get all model classes from schema.

        Returns:
            Dictionary mapping class names to model classes.
        """
        if self._schema is not None:
            return {
                "Query": self._schema.Query,
                "Chunk": self._schema.Chunk,
                "ChunkRetrievedResult": self._schema.ChunkRetrievedResult,
                "ExecutorResult": self._schema.ExecutorResult,
                "Pipeline": self._schema.Pipeline,
            }

        from autorag_research.orm.schema import (
            Chunk,
            ChunkRetrievedResult,
            ExecutorResult,
            Pipeline,
            Query,
        )

        return {
            "Query": Query,
            "Chunk": Chunk,
            "ChunkRetrievedResult": ChunkRetrievedResult,
            "ExecutorResult": ExecutorResult,
            "Pipeline": Pipeline,
        }

    def _reset_repositories(self) -> None:
        """Reset all repository references to None."""
        self._query_repo = None
        self._chunk_repo = None
        self._chunk_result_repo = None
        self._executor_result_repo = None
        self._pipeline_repo = None

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

    @property
    def executor_results(self) -> ExecutorResultRepository:
        """Get the ExecutorResult repository.

        Returns:
            ExecutorResultRepository instance.

        Raises:
            SessionNotSetError: If session is not initialized.
        """
        return self._get_repository(
            "_executor_result_repo",
            ExecutorResultRepository,
            lambda: self._get_schema_classes()["ExecutorResult"],
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
