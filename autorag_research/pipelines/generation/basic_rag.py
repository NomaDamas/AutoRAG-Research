"""Basic RAG Pipeline for AutoRAG-Research.

Implements simple single-call RAG: retrieve once -> build prompt -> generate once.
"""

from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseLanguageModel
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.config import BaseGenerationPipelineConfig
from autorag_research.orm.service.generation_pipeline import GenerationResult
from autorag_research.pipelines.generation.base import BaseGenerationPipeline
from autorag_research.pipelines.retrieval.base import BaseRetrievalPipeline
from autorag_research.util import TokenUsageTracker

DEFAULT_PROMPT_TEMPLATE = """Context:
{context}

Question: {query}

Answer:"""


@dataclass(kw_only=True)
class BasicRAGPipelineConfig(BaseGenerationPipelineConfig):
    """Configuration for a Basic RAG pipeline.

    Attributes:
        name: Unique name for this pipeline instance.
        retrieval_pipeline_name: Name of the retrieval pipeline to use.
            The Executor will automatically load and instantiate this pipeline
            from configs/pipelines/{name}.yaml.
        llm: LangChain BaseLanguageModel instance for text generation.
        prompt_template: Template for building the generation prompt.
            Must contain {context} and {query} placeholders.
        top_k: Number of chunks to retrieve per query. Default: 10.
        batch_size: Number of queries to fetch from DB at once. Default: 128.
        max_concurrency: Maximum concurrent async operations. Default: 16.
        max_retries: Maximum retry attempts for failed queries. Default: 3.
        retry_delay: Base delay (seconds) for exponential backoff. Default: 1.0.

    Example:
        ```python
        from langchain_openai import ChatOpenAI

        config = BasicRAGPipelineConfig(
            name="basic_rag_v1",
            retrieval_pipeline_name="bm25_baseline",
            llm=ChatOpenAI(model="gpt-4"),
            prompt_template="Context:\\n{context}\\n\\nQ: {query}\\nA:",
            top_k=5,
            max_concurrency=8,  # Limit parallelism to avoid rate limits
        )
        ```
    """

    prompt_template: str = field(default=DEFAULT_PROMPT_TEMPLATE)

    def get_pipeline_class(self) -> type["BasicRAGPipeline"]:
        """Return the BasicRAGPipeline class."""
        return BasicRAGPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        """Return kwargs for BasicRAGPipeline constructor."""
        if self._retrieval_pipeline is None:
            msg = f"Retrieval pipeline '{self.retrieval_pipeline_name}' not injected"
            raise ValueError(msg)
        return {
            "llm": self.llm,
            "retrieval_pipeline": self._retrieval_pipeline,
            "prompt_template": self.prompt_template,
        }


class BasicRAGPipeline(BaseGenerationPipeline):
    """Simple single-call RAG pipeline: retrieve once -> build prompt -> generate once.

    This pipeline implements the most basic RAG pattern:
    1. Take a query
    2. Retrieve relevant chunks using the composed retrieval pipeline
    3. Build a prompt with retrieved context
    4. Call LLM once to generate the answer

    The retrieval pipeline can be any BaseRetrievalPipeline implementation
    (BM25, vector search, hybrid, HyDE, etc.), providing flexibility in
    the retrieval strategy while keeping the generation simple.

    Example:
        ```python
        from langchain_openai import ChatOpenAI

        from autorag_research.orm.connection import DBConnection
        from autorag_research.pipelines.generation.basic_rag import BasicRAGPipeline
        from autorag_research.pipelines.retrieval.bm25 import BM25RetrievalPipeline

        db = DBConnection.from_config()  # or DBConnection.from_env()
        session_factory = db.get_session_factory()

        # Create retrieval pipeline
        retrieval_pipeline = BM25RetrievalPipeline(
            session_factory=session_factory,
            name="bm25_baseline",
            tokenizer="bert",
        )

        # Create generation pipeline
        pipeline = BasicRAGPipeline(
            session_factory=session_factory,
            name="basic_rag_v1",
            llm=ChatOpenAI(model="gpt-4"),
            retrieval_pipeline=retrieval_pipeline,
        )

        # Run pipeline
        results = pipeline.run(top_k=5)
        ```
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        llm: "BaseLanguageModel",
        retrieval_pipeline: "BaseRetrievalPipeline",
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        schema: Any | None = None,
    ):
        """Initialize Basic RAG pipeline.

        Args:
            session_factory: SQLAlchemy sessionmaker for database connections.
            name: Name for this pipeline.
            llm: LangChain BaseLanguageModel instance for text generation.
            retrieval_pipeline: Retrieval pipeline for fetching relevant context.
            prompt_template: Template string with {context} and {query} placeholders.
            schema: Schema namespace from create_schema(). If None, uses default schema.
        """
        # Store prompt template before calling super().__init__
        # because _get_pipeline_config() is called in super().__init__
        self._prompt_template = prompt_template

        super().__init__(session_factory, name, llm, retrieval_pipeline, schema)

    def _get_pipeline_config(self) -> dict[str, Any]:
        """Return Basic RAG pipeline configuration."""
        return {
            "type": "basic_rag",
            "prompt_template": self._prompt_template,
            "retrieval_pipeline_id": self._retrieval_pipeline.pipeline_id,
        }

    async def _generate(self, query_id: int | str, top_k: int) -> GenerationResult:
        """Generate answer using simple RAG: retrieve once, generate once (async).

        Args:
            query_id: The query ID to answer.
            top_k: Number of chunks to retrieve.

        Returns:
            GenerationResult containing the generated text and metadata.
        """
        retrieved = await self._retrieval_pipeline._retrieve_by_id(query_id, top_k)

        chunk_ids = [r["doc_id"] for r in retrieved]
        chunk_contents = self._service.get_chunk_contents(chunk_ids)

        query_text = self._service.get_query_text(query_id)

        # 3. Build prompt with context
        context = "\n\n".join(chunk_contents)
        prompt = self._prompt_template.format(context=context, query=query_text)

        # 4. Async LLM call (main I/O benefit)
        response = await self._llm.ainvoke(prompt)

        # 5. Extract token usage from response metadata
        tracker = TokenUsageTracker()
        token_usage = tracker.record(response)

        # Extract text content from response
        text = response.content if hasattr(response, "content") else str(response)

        return GenerationResult(
            text=text,
            token_usage=token_usage,
            metadata={
                "retrieved_chunk_ids": chunk_ids,
                "retrieved_scores": [r["score"] for r in retrieved],
            },
        )
