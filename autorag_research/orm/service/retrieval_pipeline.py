"""Retrieval Pipeline Service for AutoRAG-Research.

Provides service layer for running retrieval pipelines:
1. Fetch queries from database
2. Run retrieval using provided retrieval function
3. Store retrieval results (ChunkRetrievedResult)
"""

import logging
from collections.abc import Callable
from typing import Any, Literal

from sqlalchemy.orm import Session, sessionmaker

from autorag_research.orm.service.base import BaseService
from autorag_research.orm.uow.retrieval_uow import RetrievalUnitOfWork

__all__ = ["RetrievalFunc", "RetrievalPipelineService"]

logger = logging.getLogger("AutoRAG-Research")

# Type alias for retrieval function
# Input: list of query IDs, top_k
# Output: list of list of dicts with 'doc_id' and 'score'
RetrievalFunc = Callable[[list[int | str], int], list[list[dict[str, Any]]]]


class RetrievalPipelineService(BaseService):
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

        # Create service
        service = RetrievalPipelineService(session_factory, schema)

        # Direct search (for single-query use cases)
        results = service.bm25_search(query_ids=[1, 2, 3], top_k=10)
        results = service.vector_search(query_ids=[1, 2, 3], top_k=10)

        # Or use run_pipeline for batch processing with result persistence
        pipeline_id = service.save_pipeline(
            name="bm25",
            config={"type": "bm25", "tokenizer": "bert"},
        )
        stats = service.run_pipeline(
            retrieval_func=lambda ids, k: service.bm25_search(ids, k),
            pipeline_id=pipeline_id,
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
        super().__init__(session_factory, schema)

    def _get_schema_classes(self) -> dict[str, Any]:
        """Get schema classes from the schema namespace.

        Returns:
            Dictionary mapping class names to ORM classes.
        """
        if self._schema is not None:
            return {
                "Pipeline": self._schema.Pipeline,
                "ChunkRetrievedResult": self._schema.ChunkRetrievedResult,
            }
        from autorag_research.orm.schema import ChunkRetrievedResult, Pipeline

        return {
            "Pipeline": Pipeline,
            "ChunkRetrievedResult": ChunkRetrievedResult,
        }

    def _create_uow(self) -> RetrievalUnitOfWork:
        """Create a new RetrievalUnitOfWork instance."""
        return RetrievalUnitOfWork(self.session_factory, self._schema)

    def save_pipeline(self, name: str, config: dict) -> int:
        """Create a new pipeline in the database.

        Args:
            name: Name for this pipeline.
            config: Configuration dictionary for the pipeline.

        Returns:
            The pipeline ID.
        """
        with self._create_uow() as uow:
            pipeline = self._get_schema_classes()["Pipeline"](name=name, config=config)
            uow.pipelines.add(pipeline)
            uow.flush()
            pipeline_id = pipeline.id
            uow.commit()
            return pipeline_id

    def run_pipeline(
        self,
        retrieval_func: RetrievalFunc,
        pipeline_id: int,
        top_k: int = 10,
        batch_size: int = 100,
    ) -> dict[str, Any]:
        """Run retrieval pipeline for all queries.

        Args:
            retrieval_func: Function that performs retrieval.
                Signature: (query_ids: list[int | str], top_k: int) -> list[list[dict]]
                Each result dict must have 'doc_id' (int) and 'score' keys.
            pipeline_id: ID of the pipeline.
            top_k: Number of top documents to retrieve per query.
            batch_size: Number of queries to process in each batch.

        Returns:
            Dictionary with pipeline execution statistics:
            - pipeline_id: The pipeline ID
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

                # Extract query IDs
                query_ids = [q.id for q in queries]

                # Run retrieval with query IDs
                results = retrieval_func(query_ids, top_k)

                # Collect all results for batch insert
                batch_results = []
                for query_id, query_results in zip(query_ids, results, strict=True):
                    for result in query_results:
                        batch_results.append({
                            "query_id": query_id,
                            "pipeline_id": pipeline_id,
                            "chunk_id": result["doc_id"],
                            "rel_score": result["score"],
                        })

                # Batch insert all results at once
                if batch_results:
                    uow.chunk_results.bulk_insert(batch_results)
                    total_results += len(batch_results)

                total_queries += len(queries)
                offset += batch_size
                uow.commit()

                logger.info(f"Processed {total_queries} queries, stored {total_results} results")

        return {
            "pipeline_id": pipeline_id,
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

    def _make_retrieval_result(self, chunk: Any, score: float) -> dict[str, Any]:
        """Create a standardized retrieval result dictionary.

        Args:
            chunk: Chunk ORM model instance.
            score: Relevance score for this chunk.

        Returns:
            Dictionary with doc_id, score, and content keys.
        """
        return {"doc_id": chunk.id, "score": score, "content": chunk.contents}

    def bm25_search(
        self,
        query_ids: list[int | str],
        top_k: int = 10,
        tokenizer: str = "bert",
        index_name: str = "idx_chunk_bm25",
    ) -> list[list[dict[str, Any]]]:
        """Execute BM25 retrieval for given query IDs.

        Uses VectorChord-BM25 full-text search on the chunks table.

        Args:
            query_ids: List of query IDs to search for.
            top_k: Number of top results to return per query.
            tokenizer: Tokenizer to use for BM25 (default: "bert").
            index_name: Name of the BM25 index (default: "idx_chunk_bm25").

        Returns:
            List of result lists, one per query. Each result dict contains:
            - doc_id: Chunk ID
            - score: BM25 relevance score
            - content: Chunk text content

        Raises:
            ValueError: If a query ID is not found in the database.
        """
        all_results: list[list[dict[str, Any]]] = []
        with self._create_uow() as uow:
            for query_id in query_ids:
                query = uow.queries.get_by_id(query_id)
                if query is None:
                    raise ValueError(f"Query {query_id} not found")  # noqa: TRY003

                results = uow.chunks.bm25_search(
                    query_text=query.contents,
                    index_name=index_name,
                    limit=top_k,
                    tokenizer=tokenizer,
                )
                all_results.append([self._make_retrieval_result(chunk, score) for chunk, score in results])
        return all_results

    def vector_search(
        self,
        query_ids: list[int | str],
        top_k: int = 10,
        search_mode: Literal["single", "multi"] = "single",
    ) -> list[list[dict[str, Any]]]:
        """Execute vector search for given query IDs.

        Supports single-vector (cosine similarity) and multi-vector (MaxSim)
        search modes using VectorChord extension.

        Args:
            query_ids: List of query IDs to search for.
            top_k: Number of top results to return per query.
            search_mode: "single" for dense retrieval, "multi" for late interaction.

        Returns:
            List of result lists, one per query. Each result dict contains:
            - doc_id: Chunk ID
            - score: Similarity score (1 - distance for single, -distance for multi)
            - content: Chunk text content

        Raises:
            ValueError: If a query ID is not found or lacks required embeddings.
        """
        all_results: list[list[dict[str, Any]]] = []
        with self._create_uow() as uow:
            for query_id in query_ids:
                query = uow.queries.get_by_id(query_id)
                if query is None:
                    raise ValueError(f"Query {query_id} not found")  # noqa: TRY003

                if search_mode == "multi":
                    if query.embeddings is None:
                        raise ValueError(f"Query {query_id} has no multi-vector embeddings")  # noqa: TRY003
                    results = uow.chunks.maxsim_search(
                        query_vectors=list(query.embeddings),
                        vector_column="embeddings",
                        limit=top_k,
                    )
                    all_results.append([self._make_retrieval_result(chunk, -distance) for chunk, distance in results])
                else:
                    if query.embedding is None:
                        raise ValueError(f"Query {query_id} has no embedding")  # noqa: TRY003
                    results = uow.chunks.vector_search_with_scores(
                        query_vector=list(query.embedding),
                        limit=top_k,
                    )
                    all_results.append([
                        self._make_retrieval_result(chunk, 1 - distance) for chunk, distance in results
                    ])
        return all_results
