"""Base Retrieval Pipeline for AutoRAG-Research.

Provides abstract base class for all retrieval pipelines.
"""

from abc import ABC, abstractmethod
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from autorag_research.orm.repository.retrieval_uow import RetrievalSchemaProtocol
from autorag_research.orm.service.retrieval_pipeline import RetrievalPipelineService


class BaseRetrievalPipeline(ABC):
    """Abstract base class for all retrieval pipelines.

    This class provides common functionality for retrieval pipelines:
    - Service initialization
    - Pipeline creation in database
    - Abstract run method for subclasses to implement

    Subclasses must implement:
    - `_get_retrieval_func()`: Return the retrieval function to use
    - `_get_pipeline_config()`: Return the pipeline configuration dict
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        schema: RetrievalSchemaProtocol | None = None,
    ):
        """Initialize retrieval pipeline.

        Args:
            session_factory: SQLAlchemy sessionmaker for database connections.
            name: Name for this pipeline.
            schema: Schema namespace from create_schema(). If None, uses default schema.
        """
        self.session_factory = session_factory
        self.name = name
        self._schema = schema

        # Initialize service
        self._service = RetrievalPipelineService(session_factory, schema)

        # Create pipeline in DB
        self.pipeline_id = self._service.create_pipeline(
            name=name,
            config=self._get_pipeline_config(),
        )

    @abstractmethod
    def _get_pipeline_config(self) -> dict[str, Any]:
        """Return the pipeline configuration dictionary.

        Returns:
            Configuration dict to store in the pipeline table.
        """
        pass

    @abstractmethod
    def _get_retrieval_func(self) -> Any:
        """Return the retrieval function to use.

        Returns:
            A callable with signature: (queries: list[str], top_k: int) -> list[list[dict]]
        """
        pass

    def run(
        self,
        top_k: int = 10,
        batch_size: int = 100,
    ) -> dict[str, Any]:
        """Run the retrieval pipeline.

        Args:
            top_k: Number of top documents to retrieve per query.
            batch_size: Number of queries to process in each batch.

        Returns:
            Dictionary with pipeline execution statistics:
            - pipeline_id: The pipeline ID
            - total_queries: Number of queries processed
            - total_results: Number of results stored
        """
        return self._service.run_pipeline(
            retrieval_func=self._get_retrieval_func(),
            pipeline_id=self.pipeline_id,
            top_k=top_k,
            batch_size=batch_size,
        )
