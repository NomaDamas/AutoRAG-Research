"""Retrieval Pipeline Service for AutoRAG-Research.

Provides service layer for running retrieval pipelines:
1. Fetch queries from database
2. Run retrieval using provided retrieval function
3. Store retrieval results (ChunkRetrievedResult)
"""

import logging
from collections.abc import Callable
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from autorag_research.orm.repository.retrieval_uow import RetrievalSchemaProtocol, RetrievalUnitOfWork

__all__ = ["RetrievalFunc", "RetrievalPipelineService"]

logger = logging.getLogger("AutoRAG-Research")

# Type alias for retrieval function
# Input: list of query strings, top_k
# Output: list of list of dicts with 'doc_id' and 'score'
RetrievalFunc = Callable[[list[str], int], list[list[dict[str, Any]]]]


class RetrievalPipelineService:
    """Service for running retrieval pipelines.

    This service handles the common workflow for all retrieval pipelines:
    1. Create a pipeline instance
    2. Fetch queries from database
    3. Run retrieval using the provided retrieval function
    4. Store results in ChunkRetrievedResult table

    The actual retrieval logic is provided as a function parameter,
    making this service reusable for BM25, dense retrieval, hybrid, etc.

    Example:
        ```python
        from autorag_research.orm.service import RetrievalPipelineService
        from autorag_research.nodes.retrieval.bm25 import BM25Module

        # Initialize BM25 module
        bm25 = BM25Module(index_path="/path/to/index")

        # Create service
        service = RetrievalPipelineService(session_factory, schema)

        # Create pipeline
        pipeline_id = service.create_pipeline(
            name="bm25_baseline",
            config={"type": "bm25", "index_path": "/path/to/index"},
        )

        # Run pipeline with BM25 retrieval function
        results = service.run(
            retrieval_func=bm25.run,
            pipeline_id=pipeline_id,
            metric_id=1,
            top_k=10,
        )
        ```
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        schema: RetrievalSchemaProtocol | None = None,
    ):
        """Initialize retrieval pipeline service.

        Args:
            session_factory: SQLAlchemy sessionmaker for database connections.
            schema: Schema namespace from create_schema(). If None, uses default schema.
        """
        self.session_factory = session_factory
        self._schema: RetrievalSchemaProtocol = self._resolve_schema(schema)

    def _resolve_schema(self, schema: RetrievalSchemaProtocol | None) -> RetrievalSchemaProtocol:
        """Resolve schema, using default if not provided."""
        if schema is not None:
            return schema
        from autorag_research.orm import schema as default_schema

        return default_schema  # type: ignore[return-value]

    def _create_uow(self) -> RetrievalUnitOfWork:
        """Create a new RetrievalUnitOfWork instance."""
        return RetrievalUnitOfWork(self.session_factory, self._schema)

    def create_pipeline(self, name: str, config: dict) -> int:
        """Create a new pipeline in the database.

        Args:
            name: Name for this pipeline.
            config: Configuration dictionary for the pipeline.

        Returns:
            The pipeline ID.
        """
        with self._create_uow() as uow:
            pipeline = self._schema.Pipeline(name=name, config=config)
            uow.pipelines.add(pipeline)
            uow.flush()
            pipeline_id = pipeline.id
            uow.commit()
            return pipeline_id

    def run(
        self,
        retrieval_func: RetrievalFunc,
        pipeline_id: int,
        metric_id: int,
        top_k: int = 10,
        batch_size: int = 100,
    ) -> dict[str, Any]:
        """Run retrieval pipeline.

        Args:
            retrieval_func: Function that performs retrieval.
                Signature: (queries: list[str], top_k: int) -> list[list[dict]]
                Each result dict must have 'doc_id' (int) and 'score' keys.
            pipeline_id: ID of the pipeline.
            metric_id: ID of the metric to use.
            top_k: Number of top documents to retrieve per query.
            batch_size: Number of queries to process in each batch.

        Returns:
            Dictionary with pipeline execution statistics:
            - pipeline_id: The pipeline ID
            - metric_id: The metric ID
            - total_queries: Number of queries processed
            - total_results: Number of results stored
        """
        # Process queries in batches
        total_queries = 0
        total_results = 0
        offset = 0

        while True:
            with self._create_uow() as uow:
                # Fetch batch of queries
                queries = uow.queries.get_all(limit=batch_size, offset=offset)
                if not queries:
                    break

                # Extract query texts and IDs
                query_texts = [q.contents for q in queries]
                query_ids = [q.id for q in queries]

                # Run retrieval
                results = retrieval_func(query_texts, top_k)

                # Process and store results
                for query_id, query_results in zip(query_ids, results, strict=True):
                    for result in query_results:
                        chunk_id = result["doc_id"]
                        score = result["score"]

                        uow.chunk_results.add(
                            self._schema.ChunkRetrievedResult(
                                query_id=query_id,
                                pipeline_id=pipeline_id,
                                metric_id=metric_id,
                                chunk_id=chunk_id,
                                rel_score=score,
                            )
                        )
                        total_results += 1

                total_queries += len(queries)
                offset += batch_size
                uow.commit()

                logger.info(f"Processed {total_queries} queries, stored {total_results} results")

        return {
            "pipeline_id": pipeline_id,
            "metric_id": metric_id,
            "total_queries": total_queries,
            "total_results": total_results,
        }

    def get_pipeline_config(self, pipeline_id: int) -> dict | None:
        """Get pipeline configuration by ID.

        Args:
            pipeline_id: ID of the pipeline.

        Returns:
            Pipeline config dict if found, None otherwise.
        """
        with self._create_uow() as uow:
            pipeline = uow.pipelines.get_by_id(pipeline_id)
            return pipeline.config if pipeline else None

    def delete_pipeline_results(self, pipeline_id: int) -> int:
        """Delete all retrieval results for a specific pipeline.

        Args:
            pipeline_id: ID of the pipeline.

        Returns:
            Number of deleted records.
        """
        with self._create_uow() as uow:
            deleted_count = uow.chunk_results.delete_by_pipeline(pipeline_id)
            uow.commit()
            return deleted_count
