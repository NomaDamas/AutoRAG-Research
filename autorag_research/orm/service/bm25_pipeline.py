"""BM25 Pipeline Service for AutoRAG-Research.

Provides service layer for running BM25 retrieval pipeline:
1. Fetch queries from database
2. Run BM25 retrieval using provided index
3. Store retrieval results (ChunkRetrievedResult)
"""

import logging
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from autorag_research.exceptions import SessionNotSetError
from autorag_research.orm.repository.bm25_uow import BM25PipelineUnitOfWork
from autorag_research.orm.service.base import BaseService

logger = logging.getLogger("AutoRAG-Research")


class BM25PipelineService(BaseService):
    """Service for running BM25 retrieval pipeline.

    This service:
    1. Fetches queries from the database
    2. Runs BM25 retrieval using Pyserini
    3. Maps retrieved doc_ids to chunk_ids
    4. Stores results in ChunkRetrievedResult table

    Example:
        ```python
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from autorag_research.orm.service import BM25PipelineService

        # Setup database connection
        engine = create_engine("postgresql://user:pass@localhost/dbname")
        session_factory = sessionmaker(bind=engine)

        # Initialize service
        service = BM25PipelineService(session_factory)

        # Run BM25 pipeline
        results = service.run(
            index_path="/path/to/lucene/index",
            pipeline_name="bm25_baseline",
            top_k=10,
        )
        ```
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        schema: Any | None = None,
    ):
        """Initialize BM25 pipeline service.

        Args:
            session_factory: SQLAlchemy sessionmaker for database connections.
            schema: Schema namespace from create_schema(). If None, uses default 768-dim schema.
        """
        super().__init__(session_factory, schema)

    def _create_uow(self) -> BM25PipelineUnitOfWork:
        """Create a new BM25PipelineUnitOfWork instance.

        Returns:
            New BM25PipelineUnitOfWork instance.
        """
        return BM25PipelineUnitOfWork(self.session_factory, self._schema)

    def _get_schema_classes(self) -> dict[str, type]:
        """Get schema classes from the schema namespace.

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

    def run(
        self,
        index_path: str,
        pipeline_name: str,
        metric_name: str = "bm25",
        top_k: int = 10,
        k1: float = 0.9,
        b: float = 0.4,
        language: str = "en",
        batch_size: int = 100,
        doc_id_to_chunk_id: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Run BM25 retrieval pipeline.

        This method:
        1. Creates or retrieves the pipeline and metric records
        2. Fetches all queries from the database
        3. Runs BM25 retrieval in batches
        4. Maps doc_ids to chunk_ids and stores results

        Args:
            index_path: Path to the Lucene index directory.
            pipeline_name: Name for this pipeline run (used to identify results).
            metric_name: Name for the metric (default: "bm25").
            top_k: Number of top documents to retrieve per query.
            k1: BM25 k1 parameter (controls term frequency saturation).
            b: BM25 b parameter (controls length normalization).
            language: Language for analyzer ('en', 'ko', 'zh', 'ja', etc.).
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
        # Initialize BM25 module (lazy import to avoid Java dependency at import time)
        from autorag_research.nodes.retrieval.bm25 import BM25Module

        bm25 = BM25Module(
            index_path=index_path,
            k1=k1,
            b=b,
            language=language,
        )

        # Create or get pipeline
        pipeline_id = self._get_or_create_pipeline(
            name=pipeline_name,
            config={
                "type": "bm25",
                "index_path": index_path,
                "top_k": top_k,
                "k1": k1,
                "b": b,
                "language": language,
            },
        )

        # Create or get metric
        metric_id = self._get_or_create_metric(name=metric_name, metric_type="retrieval")

        # Process queries in batches
        total_queries = 0
        total_results = 0
        offset = 0

        while True:
            with self._create_uow() as uow:
                if uow.session is None:
                    raise SessionNotSetError

                # Fetch batch of queries
                queries = uow.queries.get_all(limit=batch_size, offset=offset)
                if not queries:
                    break

                # Extract query texts and IDs
                query_texts = [q.contents for q in queries]
                query_ids = [q.id for q in queries]

                # Run BM25 retrieval
                results = bm25.run(queries=query_texts, top_k=top_k)

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
                            # Assume doc_id is the chunk_id as integer
                            try:
                                chunk_id = int(doc_id)
                            except ValueError:
                                logger.warning(f"Cannot convert doc_id {doc_id} to int, skipping")
                                continue

                        # Create result record
                        uow.chunk_retrieved_results.add(
                            uow.chunk_retrieved_results.model_cls(
                                query_id=query_id,
                                pipeline_id=pipeline_id,
                                metric_id=metric_id,
                                chunk_id=chunk_id,
                                rel_score=score,
                            )
                        )
                        total_results += 1

                uow.commit()
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
        """Get existing pipeline by name or create a new one.

        Args:
            name: Pipeline name.
            config: Pipeline configuration dictionary.

        Returns:
            Pipeline ID.
        """
        with self._create_uow() as uow:
            if uow.session is None:
                raise SessionNotSetError

            # Try to find existing pipeline
            existing = uow.pipelines.get_by_name(name)
            if existing:
                return existing.id

            # Create new pipeline
            classes = self._get_schema_classes()
            pipeline = classes["Pipeline"](name=name, config=config)
            uow.pipelines.add(pipeline)
            uow.flush()
            pipeline_id = pipeline.id
            uow.commit()
            return pipeline_id

    def _get_or_create_metric(self, name: str, metric_type: str) -> int:
        """Get existing metric by name and type or create a new one.

        Args:
            name: Metric name.
            metric_type: Metric type ('retrieval' or 'generation').

        Returns:
            Metric ID.
        """
        with self._create_uow() as uow:
            if uow.session is None:
                raise SessionNotSetError

            # Try to find existing metric
            existing = uow.metrics.get_by_name_and_type(name, metric_type)
            if existing:
                return existing.id

            # Create new metric
            classes = self._get_schema_classes()
            metric = classes["Metric"](name=name, type=metric_type)
            uow.metrics.add(metric)
            uow.flush()
            metric_id = metric.id
            uow.commit()
            return metric_id

    def delete_pipeline_results(self, pipeline_id: int) -> int:
        """Delete all retrieval results for a specific pipeline.

        Args:
            pipeline_id: The pipeline ID.

        Returns:
            Number of deleted records.
        """
        with self._create_uow() as uow:
            if uow.session is None:
                raise SessionNotSetError

            deleted = uow.chunk_retrieved_results.delete_by_pipeline(pipeline_id)
            uow.commit()
            return deleted

    def get_pipeline_statistics(self, pipeline_id: int) -> dict[str, Any]:
        """Get statistics for a specific pipeline.

        Args:
            pipeline_id: The pipeline ID.

        Returns:
            Dictionary with pipeline statistics.
        """
        with self._create_uow() as uow:
            if uow.session is None:
                raise SessionNotSetError

            pipeline = uow.pipelines.get_by_id(pipeline_id)
            if not pipeline:
                return {"error": "Pipeline not found"}

            total_results = uow.chunk_retrieved_results.count()

            return {
                "pipeline_id": pipeline_id,
                "pipeline_name": pipeline.name,
                "config": pipeline.config,
                "total_results": total_results,
            }
