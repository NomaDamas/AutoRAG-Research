"""Generation Pipeline Service for AutoRAG-Research.

Provides service layer for running generation pipelines:
1. Fetch queries from database
2. Run generation using provided generate function (which uses retrieval internally)
3. Store generation results (ExecutorResult)
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from autorag_research.orm.service.base import BaseService
from autorag_research.orm.uow.generation_uow import GenerationUnitOfWork
from autorag_research.util import run_with_concurrency_limit

__all__ = ["GenerateFunc", "GenerationPipelineService", "GenerationResult"]

logger = logging.getLogger("AutoRAG-Research")


@dataclass
class GenerationResult:
    """Result of a single generation call.

    Attributes:
        text: The generated text response.
        token_usage: Detailed token breakdown as JSONB dict with keys:
            - prompt_tokens: Number of tokens in the prompt
            - completion_tokens: Number of tokens in the completion
            - total_tokens: Total tokens used
            - embedding_tokens: Number of tokens for embeddings (if any)
        metadata: Optional metadata dict (can store retrieval info, intermediate steps, etc.).
    """

    text: str
    token_usage: dict | None = None
    metadata: dict | None = None


# Type alias for async generation function - processes ONE query
# Signature: (query_id: int, top_k: int) -> Awaitable[GenerationResult]
# The function has internal access to retrieval pipeline via closure/method binding
GenerateFunc = Callable[[int, int], Awaitable[GenerationResult]]


def aggregate_batch_token_usage(results: list[dict]) -> tuple[int, int, int, int]:
    """Aggregate token usage from generation results.

    Args:
        results: List of generation result dicts with token_usage and execution_time.

    Returns:
        Tuple of (prompt_tokens, completion_tokens, embedding_tokens, execution_time_ms).
    """
    prompt_tokens = 0
    completion_tokens = 0
    embedding_tokens = 0
    execution_time_ms = 0

    for result in results:
        if result["token_usage"]:
            prompt_tokens += result["token_usage"].get("prompt_tokens", 0)
            completion_tokens += result["token_usage"].get("completion_tokens", 0)
            embedding_tokens += result["token_usage"].get("embedding_tokens", 0)
        execution_time_ms += result["execution_time"]

    return prompt_tokens, completion_tokens, embedding_tokens, execution_time_ms

class GenerationPipelineService(BaseService):
    """Service for running generation pipelines.

    This service handles the common workflow for all generation pipelines:
    1. Create a pipeline instance
    2. Fetch queries from database
    3. Run generation using the provided function (which handles retrieval internally)
    4. Store results in ExecutorResult table

    The actual generation logic (including retrieval) is provided as a function parameter,
    making this service reusable for NaiveRAG, iterative RAG, etc.

    Example:
        ```python
        from autorag_research.orm.service.generation_pipeline import GenerationPipelineService

        # Create service
        service = GenerationPipelineService(session_factory, schema)

        # Create pipeline
        pipeline_id = service.save_pipeline(
            name="naive_rag_v1",
            config={"type": "naive_rag", "llm_model": "gpt-4"},
        )

        # Run pipeline with async generation function
        results = service.run_pipeline(
            generate_func=my_async_generate_func,  # Async: handles retrieval + generation
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
        """Initialize generation pipeline service.

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
                "ExecutorResult": self._schema.ExecutorResult,
            }
        from autorag_research.orm.schema import ExecutorResult, Pipeline

        return {
            "Pipeline": Pipeline,
            "ExecutorResult": ExecutorResult,
        }

    def _create_uow(self) -> GenerationUnitOfWork:
        """Create a new GenerationUnitOfWork instance."""
        return GenerationUnitOfWork(self.session_factory, self._schema)

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

    def _filter_valid_results(
        self,
        query_ids: list[int],
        batch_results: list[dict | None],
        failed_queries: list[int],
    ) -> list[dict]:
        """Filter valid results and track failed queries."""
        valid_results = []
        for query_id, result in zip(query_ids, batch_results, strict=True):
            if result is None:
                failed_queries.append(query_id)
            else:
                valid_results.append(result)
        return valid_results

    def _save_executor_results(self, uow: GenerationUnitOfWork, valid_results: list[dict]) -> None:
        """Save executor results to database."""
        if valid_results:
            executor_result_class = self._get_schema_classes()["ExecutorResult"]
            entities = [executor_result_class(**item) for item in valid_results]
            uow.executor_results.add_all(entities)

    def run_pipeline(
        self,
        generate_func: GenerateFunc,
        pipeline_id: int,
        top_k: int = 10,
        batch_size: int = 128,
        max_concurrency: int = 16,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> dict[str, Any]:
        """Run generation pipeline for all queries with parallel execution and retry.

        Args:
            generate_func: Async function that performs retrieval + generation.
                Signature: async (query_id: int, top_k: int) -> GenerationResult
                The function should handle retrieval internally.
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
            - token_usage: Aggregated token usage dict (prompt_tokens, completion_tokens, total_tokens, embedding_tokens)
            - avg_execution_time_ms: Average execution time per query
            - failed_queries: List of query IDs that failed after all retries
        """
        from tenacity import (
            AsyncRetrying,
            RetryError,
            stop_after_attempt,
            wait_exponential,
        )

        total_queries = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_embedding_tokens = 0
        total_execution_time_ms = 0
        failed_queries: list[int] = []
        offset = 0

        async def process_query_with_retry(query_id: int) -> dict | None:
            """Process a single query with retry logic."""
            start_time = time.time()
            try:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(max_retries),
                    wait=wait_exponential(multiplier=retry_delay, min=retry_delay, max=60),
                    reraise=True,
                ):
                    with attempt:
                        result = await generate_func(query_id, top_k)
                        execution_time_ms = int((time.time() - start_time) * 1000)
                        return {
                            "query_id": query_id,
                            "pipeline_id": pipeline_id,
                            "generation_result": result.text,
                            "token_usage": result.token_usage,
                            "execution_time": execution_time_ms,
                            "result_metadata": result.metadata,
                        }
            except RetryError:
                logger.exception(f"Generation failed for query {query_id} after {max_retries} attempts")
            except Exception:
                logger.exception(f"Generation failed for query {query_id}")
            return None

        async def process_batch(query_ids: list[int]) -> list[dict | None]:
            """Process a batch of queries with concurrency limit."""
            return await run_with_concurrency_limit(
                items=query_ids,
                async_func=process_query_with_retry,
                max_concurrency=max_concurrency,
                error_message="Generation failed",
            )

        while True:
            with self._create_uow() as uow:
                queries = uow.queries.get_all(limit=batch_size, offset=offset)
                if not queries:
                    break

                query_ids = [q.id for q in queries]
                batch_results = asyncio.run(process_batch(query_ids))
                valid_results = self._filter_valid_results(query_ids, batch_results, failed_queries)

                prompt, completion, embedding, exec_time = aggregate_batch_token_usage(valid_results)
                total_prompt_tokens += prompt
                total_completion_tokens += completion
                total_embedding_tokens += embedding
                total_execution_time_ms += exec_time

                self._save_executor_results(uow, valid_results)
                total_queries += len(valid_results)
                offset += batch_size
                uow.commit()

                logger.info(f"Processed {total_queries} queries")

        if failed_queries:
            logger.warning(f"Failed to process {len(failed_queries)} queries after retries")

        avg_execution_time_ms = total_execution_time_ms / total_queries if total_queries > 0 else 0

        # Build aggregated token usage
        total_tokens = total_prompt_tokens + total_completion_tokens
        token_usage = None
        if total_tokens > 0:
            token_usage = {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens,
                "embedding_tokens": total_embedding_tokens,
            }

        return {
            "pipeline_id": pipeline_id,
            "total_queries": total_queries,
            "token_usage": token_usage,
            "avg_execution_time_ms": avg_execution_time_ms,
            "failed_queries": failed_queries,
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
        """Delete all generation results for a specific pipeline.

        Args:
            pipeline_id: ID of the pipeline.

        Returns:
            Number of deleted records.
        """
        with self._create_uow() as uow:
            deleted_count = uow.executor_results.delete_by_pipeline(pipeline_id)
            uow.commit()
            return deleted_count

    def get_chunk_contents(self, chunk_ids: list[int | str]) -> list[str]:
        """Get chunk contents by IDs.

        Args:
            chunk_ids: List of chunk IDs to fetch.

        Returns:
            List of chunk content strings in the same order as input IDs.
        """
        with self._create_uow() as uow:
            chunks = uow.chunks.get_by_ids(chunk_ids)
            # Create a map for preserving order
            chunk_map = {chunk.id: chunk.contents for chunk in chunks}
            return [chunk_map.get(cid, "") for cid in chunk_ids]
