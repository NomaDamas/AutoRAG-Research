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
    - `_generate()`: Generate an answer given a query (has access to a retrieval pipeline)
    - `_get_pipeline_config()`: Return the pipeline configuration dict

    Example:
        ```python
        class BasicRAGPipeline(BaseGenerationPipeline):
            def _generate(self, query: str, top_k: int) -> GenerationResult:
                # Retrieve relevant chunks
                results = self._retrieval_pipeline.retrieve(query, top_k)
                chunks = [self._get_chunk_content(r["doc_id"]) for r in results]

                # Build prompt and generate
                context = "\\n\\n".join(chunks)
                prompt = f"Context:\\n{context}\\n\\nQuestion: {query}\\n\\nAnswer:"
                response = self._llm.invoke(prompt)

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

    @abstractmethod
    def _generate(self, query: str, top_k: int) -> GenerationResult:
        """Generate an answer for a query.

        This method has full access to self._retrieval_pipeline for:
        - Single retrieval: self._retrieval_pipeline.retrieve(query, top_k)
        - Multiple retrievals: call retrieve() in a loop
        - Query rewriting: modify query, retrieve again
        - Agent-style: decide next action based on results

        Subclasses implement their generation strategy.

        Args:
            query: The query text to answer.
            top_k: Number of chunks to retrieve.

        Returns:
            GenerationResult containing the generated text and optional metadata.
        """
        pass

    def run(
        self,
        top_k: int = 10,
        batch_size: int = 100,
    ) -> dict[str, Any]:
        """Run the generation pipeline.

        Args:
            top_k: Number of top documents to retrieve per query.
            batch_size: Number of queries to process in each batch.

        Returns:
            Dictionary with pipeline execution statistics:
            - pipeline_id: The pipeline ID
            - total_queries: Number of queries processed
            - total_tokens: Total tokens used (if available)
            - avg_execution_time_ms: Average execution time per query
        """
        return self._service.run_pipeline(
            generate_func=self._generate,
            pipeline_id=self.pipeline_id,
            top_k=top_k,
            batch_size=batch_size,
        )
