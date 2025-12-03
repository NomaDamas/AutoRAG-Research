"""Retrieval Pipeline Service for AutoRAG-Research.

Provides service layer for running retrieval pipelines:
1. Fetch queries from database
2. Run retrieval using provided retrieval function
3. Store retrieval results (ChunkRetrievedResult)
"""

import logging
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from autorag_research.orm.repository.chunk_retrieved_result import ChunkRetrievedResultRepository
from autorag_research.orm.repository.metric import MetricRepository
from autorag_research.orm.repository.pipeline import PipelineRepository
from autorag_research.orm.repository.query import QueryRepository

logger = logging.getLogger("AutoRAG-Research")

# Type alias for retrieval function
# Input: list of query strings, top_k
# Output: list of list of dicts with 'doc_id' and 'score'
RetrievalFunc = Callable[[list[str], int], list[list[dict[str, Any]]]]


class RetrievalPipelineService:
    """Service for running retrieval pipelines.

    This service handles the common workflow for all retrieval pipelines:
    1. Fetch queries from database
    2. Run retrieval using the provided retrieval function
    3. Map doc_ids to chunk_ids
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
        service = RetrievalPipelineService(session_factory)

        # Run pipeline with BM25 retrieval function
        results = service.run(
            retrieval_func=bm25.run,
            pipeline_name="bm25_baseline",
            pipeline_config={"type": "bm25", "index_path": "/path/to/index"},
            top_k=10,
        )
        ```
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        schema: Any | None = None,
    ):
        """Initialize retrieval pipeline service.

        Args:
            session_factory: SQLAlchemy sessionmaker for database connections.
            schema: Schema namespace from create_schema(). If None, uses default schema.
        """
        self.session_factory = session_factory
        self._schema = schema

    def _get_schema_classes(self) -> dict[str, type]:
        """Get schema classes from the schema namespace."""
        if self._schema is not None:
            return {
                "Query": self._schema.Query,
                "Pipeline": self._schema.Pipeline,
                "Metric": self._schema.Metric,
                "ChunkRetrievedResult": self._schema.ChunkRetrievedResult,
            }
        from autorag_research.orm.schema import ChunkRetrievedResult, Metric, Pipeline, Query

        return {
            "Query": Query,
            "Pipeline": Pipeline,
            "Metric": Metric,
            "ChunkRetrievedResult": ChunkRetrievedResult,
        }

    @contextmanager
    def _session_scope(self):
        """Provide a transactional scope around a series of operations."""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _get_query_repo(self, session: Session) -> QueryRepository:
        """Get Query repository for the given session."""
        classes = self._get_schema_classes()
        return QueryRepository(session, classes["Query"])

    def _get_pipeline_repo(self, session: Session) -> PipelineRepository:
        """Get Pipeline repository for the given session."""
        return PipelineRepository(session)

    def _get_metric_repo(self, session: Session) -> MetricRepository:
        """Get Metric repository for the given session."""
        return MetricRepository(session)

    def _get_chunk_retrieved_result_repo(self, session: Session) -> ChunkRetrievedResultRepository:
        """Get ChunkRetrievedResult repository for the given session."""
        classes = self._get_schema_classes()
        return ChunkRetrievedResultRepository(session, classes["ChunkRetrievedResult"])

    def run(
        self,
        retrieval_func: RetrievalFunc,
        pipeline_name: str,
        pipeline_config: dict[str, Any],
        metric_name: str = "retrieval",
        top_k: int = 10,
        batch_size: int = 100,
        doc_id_to_chunk_id: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Run retrieval pipeline.

        Args:
            retrieval_func: Function that performs retrieval.
                Signature: (queries: list[str], top_k: int) -> list[list[dict]]
                Each result dict must have 'doc_id' and 'score' keys.
            pipeline_name: Name for this pipeline run.
            pipeline_config: Configuration dictionary for the pipeline.
            metric_name: Name for the metric (default: "retrieval").
            top_k: Number of top documents to retrieve per query.
            batch_size: Number of queries to process in each batch.
            doc_id_to_chunk_id: Optional mapping from document IDs to chunk IDs.
                If not provided, assumes doc_id equals chunk_id (as integer).

        Returns:
            Dictionary with pipeline execution statistics:
            - pipeline_id: The pipeline ID
            - metric_id: The metric ID
            - total_queries: Number of queries processed
            - total_results: Number of results stored
        """
        # Create or get pipeline and metric
        pipeline_id = self._get_or_create_pipeline(name=pipeline_name, config=pipeline_config)
        metric_id = self._get_or_create_metric(name=metric_name, metric_type="retrieval")

        # Process queries in batches
        total_queries = 0
        total_results = 0
        offset = 0
        classes = self._get_schema_classes()

        while True:
            with self._session_scope() as session:
                query_repo = self._get_query_repo(session)
                result_repo = self._get_chunk_retrieved_result_repo(session)

                # Fetch batch of queries
                queries = query_repo.get_all(limit=batch_size, offset=offset)
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
                        doc_id = result["doc_id"]
                        score = result["score"]

                        # Map doc_id to chunk_id
                        if doc_id_to_chunk_id is not None:
                            chunk_id = doc_id_to_chunk_id.get(doc_id)
                            if chunk_id is None:
                                logger.warning(f"doc_id {doc_id} not found in mapping, skipping")
                                continue
                        else:
                            try:
                                chunk_id = int(doc_id)
                            except ValueError:
                                logger.warning(f"Cannot convert doc_id {doc_id} to int, skipping")
                                continue

                        # Create result record
                        result_repo.add(
                            classes["ChunkRetrievedResult"](
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

                logger.info(f"Processed {total_queries} queries, stored {total_results} results")

        return {
            "pipeline_id": pipeline_id,
            "metric_id": metric_id,
            "total_queries": total_queries,
            "total_results": total_results,
        }

    def _get_or_create_pipeline(self, name: str, config: dict) -> int:
        """Get existing pipeline by name or create a new one."""
        with self._session_scope() as session:
            pipeline_repo = self._get_pipeline_repo(session)
            existing = pipeline_repo.get_by_name(name)
            if existing:
                return existing.id

            classes = self._get_schema_classes()
            pipeline = classes["Pipeline"](name=name, config=config)
            pipeline_repo.add(pipeline)
            session.flush()
            return pipeline.id

    def _get_or_create_metric(self, name: str, metric_type: str) -> int:
        """Get existing metric by name and type or create a new one."""
        with self._session_scope() as session:
            metric_repo = self._get_metric_repo(session)
            existing = metric_repo.get_by_name_and_type(name, metric_type)
            if existing:
                return existing.id

            classes = self._get_schema_classes()
            metric = classes["Metric"](name=name, type=metric_type)
            metric_repo.add(metric)
            session.flush()
            return metric.id

    def delete_pipeline_results(self, pipeline_id: int) -> int:
        """Delete all retrieval results for a specific pipeline."""
        with self._session_scope() as session:
            result_repo = self._get_chunk_retrieved_result_repo(session)
            return result_repo.delete_by_pipeline(pipeline_id)
