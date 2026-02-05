"""Base Retrieval Pipeline for AutoRAG-Research.

Provides abstract base class for all retrieval pipelines.
"""

from abc import ABC, abstractmethod
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from autorag_research.orm.service.retrieval_pipeline import RetrievalPipelineService
from autorag_research.pipelines.base import BasePipeline


class BaseRetrievalPipeline(BasePipeline, ABC):
    """Abstract base class for all retrieval pipelines.

    This class provides common functionality for retrieval pipelines:
    - Service initialization
    - Pipeline creation in database
    - Abstract retrieve methods for subclasses to implement

    Subclasses must implement:
    - `_retrieve_by_id()`: Async method for retrieval using query ID (query exists in DB)
    - `_retrieve_by_text()`: Async method for retrieval using raw query text
    - `_get_pipeline_config()`: Return the pipeline configuration dict
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        schema: Any | None = None,
    ):
        """Initialize retrieval pipeline.

        Args:
            session_factory: SQLAlchemy sessionmaker for database connections.
            name: Name for this pipeline.
            schema: Schema namespace from create_schema(). If None, uses default schema.
        """
        super().__init__(session_factory, name, schema)

        # Initialize service
        self._service = RetrievalPipelineService(session_factory, schema)

        # Create pipeline in DB
        self.pipeline_id = self._service.save_pipeline(
            name=name,
            config=self._get_pipeline_config(),
        )

    @abstractmethod
    async def _retrieve_by_id(self, query_id: int | str, top_k: int) -> list[dict[str, Any]]:
        """Retrieve documents using query ID (query must exist in DB).

        This method is used for batch processing where queries already exist
        in the database with pre-computed embeddings.

        Args:
            query_id: The query ID to retrieve for.
            top_k: Number of top documents to retrieve.

        Returns:
            List of result dicts containing:
            - doc_id: Chunk ID
            - score: Relevance score
            - content: Chunk text content (optional)
        """
        pass

    @abstractmethod
    async def _retrieve_by_text(self, query_text: str, top_k: int) -> list[dict[str, Any]]:
        """Retrieve documents using raw query text (may trigger embedding).

        This method is used for ad-hoc retrieval where the query doesn't exist
        in the database. Implementations may need to compute embeddings on-the-fly.

        Args:
            query_text: The query text to retrieve for.
            top_k: Number of top documents to retrieve.

        Returns:
            List of result dicts containing:
            - doc_id: Chunk ID
            - score: Relevance score
            - content: Chunk text content (optional)
        """
        pass

    async def retrieve(self, query_text: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Retrieve chunks for a single query (async).

        This method provides single-query retrieval, designed for use within
        GenerationPipeline where queries are processed one at a time.

        Checks if query exists in DB:
        - If exists: uses _retrieve_by_id() (faster, uses stored embedding)
        - If not: uses _retrieve_by_text() (may trigger embedding computation)

        Args:
            query_text: The query text to retrieve for.
            top_k: Number of chunks to retrieve.

        Returns:
            List of dicts with 'doc_id' (chunk ID) and 'score' keys.
        """
        # Check if query exists using service
        query = self._service.find_query_by_text(query_text)

        if query is not None:
            # Query exists - use ID-based retrieval (faster, uses stored embedding)
            return await self._retrieve_by_id(query.id, top_k)
        else:
            # Query doesn't exist - use text-based retrieval (may trigger embedding)
            return await self._retrieve_by_text(query_text, top_k)

    def run(
        self,
        top_k: int = 10,
        batch_size: int = 128,
        max_concurrency: int = 16,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> dict[str, Any]:
        """Run the retrieval pipeline.

        Args:
            top_k: Number of top documents to retrieve per query.
            batch_size: Number of queries to fetch from DB at once.
            max_concurrency: Maximum number of concurrent async operations.
            max_retries: Maximum number of retry attempts for failed queries.
            retry_delay: Base delay in seconds for exponential backoff between retries.

        Returns:
            Dictionary with pipeline execution statistics:
            - pipeline_id: The pipeline ID
            - total_queries: Number of queries processed successfully
            - total_results: Number of results stored
            - failed_queries: List of query IDs that failed after all retries
        """
        return self._service.run_pipeline(
            retrieval_func=self._retrieve_by_id,  # Use ID-based for batch processing
            pipeline_id=self.pipeline_id,
            top_k=top_k,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
