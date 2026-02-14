"""Retrieval Pipeline Service for AutoRAG-Research.

Provides service layer for running retrieval pipelines:
1. Fetch queries from database
2. Run retrieval using provided retrieval function
3. Store retrieval results (ChunkRetrievedResult)
"""

import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import Any, Literal

from sqlalchemy.orm import Session, sessionmaker

from autorag_research.orm.service.base import BaseService
from autorag_research.orm.uow.retrieval_uow import RetrievalUnitOfWork

__all__ = ["RetrievalFunc", "RetrievalPipelineService"]

logger = logging.getLogger("AutoRAG-Research")

# Type alias for async retrieval function - processes ONE query
# Input: query_id (int or str), top_k
# Output: list of dicts with 'doc_id' and 'score'
RetrievalFunc = Callable[[int | str, int], Coroutine[Any, Any, list[dict[str, Any]]]]


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
        pipeline_id, is_new = service.get_or_create_pipeline(
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

    def get_or_create_pipeline(self, name: str, config: dict) -> tuple[int | str, bool]:
        """Get existing pipeline by name or create a new one.

        If a pipeline with the given name already exists, returns its ID.
        If the existing pipeline has a different config, logs a warning.
        If no pipeline exists, creates a new one.

        Args:
            name: Name for this pipeline (used as experiment identifier).
            config: Configuration dictionary for the pipeline.

        Returns:
            Tuple of (pipeline_id, is_new) where is_new is True if a new pipeline was created.
        """
        with self._create_uow() as uow:
            existing = uow.pipelines.get_by_name(name)
            if existing is not None:
                if existing.config != config:
                    logger.warning(
                        f"Pipeline '{name}' exists with different config. "
                        f"Existing: {existing.config}, New: {config}. Reusing existing pipeline."
                    )
                else:
                    logger.info(f"Resuming pipeline '{name}' (pipeline_id={existing.id})")
                return existing.id, False

            pipeline = self._get_schema_classes()["Pipeline"](name=name, config=config)
            uow.pipelines.add(pipeline)
            uow.flush()
            pipeline_id = pipeline.id
            uow.commit()
            logger.info(f"Created new pipeline '{name}' (pipeline_id={pipeline_id})")
            return pipeline_id, True

    def _collect_retrieval_results(
        self,
        query_ids: list[int | str],
        results: list[list[dict] | None],
        pipeline_id: int | str,
        failed_queries: list[int | str],
    ) -> list[dict]:
        """Collect valid retrieval results and track failed queries.

        Args:
            query_ids: List of query IDs that were processed.
            results: Results from batch processing (None for failed queries).
            pipeline_id: Pipeline ID for result records.
            failed_queries: List to append failed query IDs to (mutated).

        Returns:
            List of result dicts ready for batch insert.
        """
        batch_results = []
        for query_id, query_results in zip(query_ids, results, strict=True):
            if query_results is None:
                failed_queries.append(query_id)
                continue
            for result in query_results:
                batch_results.append({
                    "query_id": query_id,
                    "pipeline_id": pipeline_id,
                    "chunk_id": result["doc_id"],
                    "rel_score": result["score"],
                })
        return batch_results

    def run_pipeline(  # noqa: C901
        self,
        retrieval_func: RetrievalFunc,
        pipeline_id: int | str,
        top_k: int = 10,
        batch_size: int = 128,
        max_concurrency: int = 16,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> dict[str, Any]:
        """Run retrieval pipeline for all queries with parallel execution and retry.

        Args:
            retrieval_func: Async function that performs retrieval for a single query.
                Signature: async (query_id: int | str, top_k: int) -> list[dict]
                Each result dict must have 'doc_id' (int) and 'score' keys.
            pipeline_id: ID of the pipeline.
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
        from tenacity import AsyncRetrying, RetryError, stop_after_attempt, wait_exponential

        from autorag_research.util import run_with_concurrency_limit

        async def process_query_with_retry(query_id: int | str) -> list[dict] | None:
            """Process a single query with retry logic."""
            try:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(max_retries),
                    wait=wait_exponential(multiplier=retry_delay, min=retry_delay, max=60),
                    reraise=True,
                ):
                    with attempt:
                        return await retrieval_func(query_id, top_k)
            except RetryError:
                logger.exception(f"Retrieval failed for query {query_id} after {max_retries} attempts")
            except Exception:
                logger.exception(f"Retrieval failed for query {query_id}")
            return None

        async def process_batch(query_ids: list[int | str]) -> list[list[dict] | None]:
            """Process a batch of queries with concurrency limit."""
            return await run_with_concurrency_limit(
                items=query_ids,
                async_func=process_query_with_retry,
                max_concurrency=max_concurrency,
                error_message="Retrieval failed",
            )

        total_queries = 0
        total_results = 0
        failed_queries: list[int | str] = []
        offset = 0

        while True:
            with self._create_uow() as uow:
                queries = uow.queries.get_all(limit=batch_size, offset=offset)
                if not queries:
                    break

                query_ids = [q.id for q in queries]

                # Skip queries that already have results (resume support)
                existing_results = uow.chunk_results.get_by_query_and_pipeline(query_ids, pipeline_id)
                completed_ids = {r.query_id for r in existing_results}
                query_ids = [qid for qid in query_ids if qid not in completed_ids]

                if not query_ids:
                    offset += batch_size
                    continue

                results = asyncio.run(process_batch(query_ids))

                batch_results = self._collect_retrieval_results(query_ids, results, pipeline_id, failed_queries)

                if batch_results:
                    uow.chunk_results.bulk_insert(batch_results)
                    total_results += len(batch_results)

                total_queries += len([r for r in results if r is not None])
                offset += batch_size
                uow.commit()

                logger.info(f"Processed {total_queries} queries, stored {total_results} results")

        if failed_queries:
            logger.warning(f"Failed to process {len(failed_queries)} queries after retries: {failed_queries}")

        return {
            "pipeline_id": pipeline_id,
            "total_queries": total_queries,
            "total_results": total_results,
            "failed_queries": failed_queries,
        }

    def get_pipeline_config(self, pipeline_id: int | str) -> dict | None:
        """Get pipeline configuration by ID.

        Args:
            pipeline_id: ID of the pipeline.

        Returns:
            Pipeline config dict if found, None otherwise.
        """
        with self._create_uow() as uow:
            pipeline = uow.pipelines.get_by_id(pipeline_id)
            return pipeline.config if pipeline else None

    def delete_pipeline_results(self, pipeline_id: int | str) -> int:
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

    def find_query_by_text(self, query_text: str) -> Any | None:
        """Find existing query by text content.

        Args:
            query_text: The query text to search for.

        Returns:
            The query if found, None otherwise.
        """
        with self._create_uow() as uow:
            return uow.queries.find_by_contents(query_text)

    def bm25_search_by_text(
        self,
        query_text: str,
        top_k: int = 10,
        tokenizer: str = "bert",
        index_name: str = "idx_chunk_bm25",
    ) -> list[dict[str, Any]]:
        """Execute BM25 retrieval using raw query text (no Query entity needed).

        Args:
            query_text: The query text to search for.
            top_k: Number of top results to return.
            tokenizer: Tokenizer to use for BM25 (default: "bert").
            index_name: Name of the BM25 index (default: "idx_chunk_bm25").

        Returns:
            List of result dicts containing doc_id, score, and content.
        """
        with self._create_uow() as uow:
            results = uow.chunks.bm25_search(
                query_text=query_text,
                index_name=index_name,
                limit=top_k,
                tokenizer=tokenizer,
            )
            return [self._make_retrieval_result(chunk, score) for chunk, score in results]

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
            - score: Relevance score in [-1, 1] range (higher = more relevant)
                - single: 1 - cosine_distance (= cosine_similarity)
                - multi: MaxSim / n_query_vectors (normalized late interaction)
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
                    query_vectors = list(query.embeddings)
                    n_query_vectors = len(query_vectors)
                    results = uow.chunks.maxsim_search(
                        query_vectors=query_vectors,
                        vector_column="embeddings",
                        limit=top_k,
                    )
                    # Normalize by number of query vectors to get [-1, 1] range
                    all_results.append([
                        self._make_retrieval_result(chunk, -distance / n_query_vectors) for chunk, distance in results
                    ])
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

    def vector_search_by_embedding(
        self,
        embedding: list[float],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Execute vector search using a provided embedding directly.

        This method enables retrieval pipelines that generate embeddings
        dynamically (like HyDE) rather than using pre-computed query embeddings.

        Args:
            embedding: The embedding vector to search with.
            top_k: Number of top results to return.

        Returns:
            List of result dicts containing doc_id, score, and content.
            Score is cosine similarity in [-1, 1] range.
        """
        with self._create_uow() as uow:
            results = uow.chunks.vector_search_with_scores(
                query_vector=embedding,
                limit=top_k,
            )
            return [self._make_retrieval_result(chunk, 1 - distance) for chunk, distance in results]

    def fetch_query_texts(self, query_ids: list[int | str]) -> list[str]:
        """Batch fetch query texts from database.

        Args:
            query_ids: List of query IDs to fetch.

        Returns:
            List of query text contents in the same order as query_ids.

        Raises:
            ValueError: If a query ID is not found.
        """
        query_texts: list[str] = []
        with self._create_uow() as uow:
            for query_id in query_ids:
                query = uow.queries.get_by_id(query_id)
                if query is None:
                    raise ValueError(f"Query {query_id} not found")  # noqa: TRY003
                query_texts.append(query.contents)
        return query_texts
