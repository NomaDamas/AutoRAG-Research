"""Base Generation Pipeline for AutoRAG-Research.

Provides abstract base class for all generation pipelines using composition
with retrieval pipelines.
"""

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models import BaseLanguageModel
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.orm.service.generation_pipeline import GenerationPipelineService, GenerationResult
from autorag_research.pipelines.base import BasePipeline
from autorag_research.pipelines.retrieval.base import BaseRetrievalPipeline


class BaseGenerationPipeline(BasePipeline, ABC):
    """Abstract base class for all generation pipelines.

    This class provides common functionality for generation pipelines:
    - Composition with a retrieval pipeline for flexible retrieval strategies
    - Service initialization for database operations
    - Pipeline creation in database
    - Abstract generate method for subclasses to implement

    Subclasses must implement:
    - `_generate()`: Async generate method given a query ID (has access to a retrieval pipeline)
    - `_get_pipeline_config()`: Return the pipeline configuration dict

    Example:
        ```python
        class BasicRAGPipeline(BaseGenerationPipeline):
            async def _generate(self, query_id: int, top_k: int) -> GenerationResult:
                # Retrieve relevant chunks by query_id (async)
                results = await self._retrieval_pipeline._retrieve_by_id(query_id, top_k)
                chunk_ids = [r["doc_id"] for r in results]
                chunk_contents = self._service.get_chunk_contents(chunk_ids)

                # Get query text (uses query_to_llm if available, else contents)
                query_text = self._get_query_text(query_id)

                # Build prompt and generate (async)
                context = "\\n\\n".join(chunk_contents)
                prompt = f"Context:\\n{context}\\n\\nQuestion: {query_text}\\n\\nAnswer:"
                response = await self._llm.ainvoke(prompt)

                return GenerationResult(text=str(response.content))
        ```
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        llm: "BaseLanguageModel",
        retrieval_pipeline: "BaseRetrievalPipeline",
        schema: Any | None = None,
    ):
        """Initialize generation pipeline.

        Args:
            session_factory: SQLAlchemy sessionmaker for database connections.
            name: Name for this pipeline.
            llm: LangChain BaseLanguageModel instance for text generation.
            retrieval_pipeline: Retrieval pipeline instance for fetching relevant context.
            schema: Schema namespace from create_schema(). If None, uses default schema.
        """
        super().__init__(session_factory, name, schema)

        self._llm = llm
        self._retrieval_pipeline = retrieval_pipeline

        # Initialize service
        self._service = GenerationPipelineService(session_factory, schema)

        # Create pipeline in DB
        self.pipeline_id = self._service.save_pipeline(
            name=name,
            config=self._get_pipeline_config(),
        )

    def _get_query_text(self, query_id: int | str) -> str:
        """Get query text by ID from database.

        Args:
            query_id: The query ID.

        Returns:
            The query text content.

        Raises:
            ValueError: If query not found.
        """
        from autorag_research.orm.uow.generation_uow import GenerationUnitOfWork

        with GenerationUnitOfWork(self.session_factory, self._schema) as uow:
            query = uow.queries.get_by_id(query_id)
            if query is None:
                raise ValueError(f"Query {query_id} not found")  # noqa: TRY003
            return query.contents

    @abstractmethod
    async def _generate(self, query_id: int | str, top_k: int) -> GenerationResult:
        """Generate an answer for a query (async).

        This method has full access to retrieval pipelines.

        Use self._get_query_text(query_id) to get the query text.
        Use self._llm.ainvoke() for async LLM calls.

        Subclasses implement their generation strategy.

        Args:
            query_id: The query ID to answer.
            top_k: Number of chunks to retrieve.

        Returns:
            GenerationResult containing the generated text and optional metadata.
        """
        pass

    def run(
        self,
        top_k: int = 10,
        batch_size: int = 128,
        max_concurrency: int = 16,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> dict[str, Any]:
        """Run the generation pipeline.

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
            - total_tokens: Total tokens used (if available)
            - avg_execution_time_ms: Average execution time per query
            - failed_queries: List of query IDs that failed after all retries
        """
        return self._service.run_pipeline(
            generate_func=self._generate,
            pipeline_id=self.pipeline_id,
            top_k=top_k,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
