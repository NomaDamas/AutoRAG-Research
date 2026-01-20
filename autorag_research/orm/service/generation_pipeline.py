"""Generation Pipeline Service for AutoRAG-Research.

Provides service layer for running generation pipelines:
1. Fetch queries from database
2. Run generation using provided generate function (which uses retrieval internally)
3. Store generation results (ExecutorResult)
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from autorag_research.orm.service.base import BaseService
from autorag_research.orm.uow.generation_uow import GenerationUnitOfWork

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


# Type alias for generation function
# Signature: (query_text: str, top_k: int) -> GenerationResult
# The function has internal access to retrieval pipeline via closure/method binding
GenerateFunc = Callable[[str, int], GenerationResult]


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

        # Run pipeline with generation function
        results = service.run_pipeline(
            generate_func=my_generate_func,  # Handles retrieval + generation
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

    def run_pipeline(
        self,
        generate_func: GenerateFunc,
        pipeline_id: int,
        top_k: int = 10,
        batch_size: int = 100,
    ) -> dict[str, Any]:
        """Run generation pipeline for all queries.

        Args:
            generate_func: Function that performs retrieval + generation.
                Signature: (query_text: str, top_k: int) -> GenerationResult
                The function should handle retrieval internally.
            pipeline_id: ID of the pipeline.
            top_k: Number of top documents to retrieve per query.
            batch_size: Number of queries to process in each batch.

        Returns:
            Dictionary with pipeline execution statistics:
            - pipeline_id: The pipeline ID
            - total_queries: Number of queries processed
            - token_usage: Aggregated token usage dict (prompt_tokens, completion_tokens, total_tokens, embedding_tokens)
            - avg_execution_time_ms: Average execution time per query
        """
        total_queries = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_embedding_tokens = 0
        total_execution_time_ms = 0
        offset = 0

        while True:
            with self._create_uow() as uow:
                # Fetch batch of queries
                queries = uow.queries.get_all(limit=batch_size, offset=offset)
                if not queries:
                    break

                # Process each query
                batch_results = []
                for query in queries:
                    # Time the generation
                    start_time = time.time()
                    result = generate_func(query.contents, top_k)
                    execution_time_ms = int((time.time() - start_time) * 1000)

                    batch_results.append({
                        "query_id": query.id,
                        "pipeline_id": pipeline_id,
                        "generation_result": result.text,
                        "token_usage": result.token_usage,
                        "execution_time": execution_time_ms,
                        "result_metadata": result.metadata,
                    })

                    # Aggregate token usage from dict
                    if result.token_usage:
                        total_prompt_tokens += result.token_usage.get("prompt_tokens", 0)
                        total_completion_tokens += result.token_usage.get("completion_tokens", 0)
                        total_embedding_tokens += result.token_usage.get("embedding_tokens", 0)
                    total_execution_time_ms += execution_time_ms

                # Batch insert executor results
                if batch_results:
                    executor_result_class = self._get_schema_classes()["ExecutorResult"]
                    entities = [executor_result_class(**item) for item in batch_results]
                    uow.executor_results.add_all(entities)

                total_queries += len(queries)
                offset += batch_size
                uow.commit()

                logger.info(f"Processed {total_queries} queries")

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
