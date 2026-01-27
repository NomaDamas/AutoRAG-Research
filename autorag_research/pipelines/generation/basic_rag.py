"""Basic RAG Pipeline for AutoRAG-Research.

Implements simple single-call RAG: retrieve once -> build prompt -> generate once.
"""

from dataclasses import dataclass, field
from typing import Any

from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.llms import LLM
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.config import BaseGenerationPipelineConfig
from autorag_research.orm.service.generation_pipeline import GenerationResult
from autorag_research.orm.uow.generation_uow import GenerationUnitOfWork
from autorag_research.pipelines.generation.base import BaseGenerationPipeline
from autorag_research.pipelines.retrieval.base import BaseRetrievalPipeline

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
        llm: LlamaIndex LLM instance for text generation.
        prompt_template: Template for building the generation prompt.
            Must contain {context} and {query} placeholders.
        top_k: Number of chunks to retrieve per query.
        batch_size: Number of queries to process in each batch.

    Example:
        ```python
        from llama_index.llms.openai import OpenAI

        config = BasicRAGPipelineConfig(
            name="basic_rag_v1",
            retrieval_pipeline_name="bm25_baseline",
            llm=OpenAI(model="gpt-4"),
            prompt_template="Context:\\n{context}\\n\\nQ: {query}\\nA:",
            top_k=5,
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
        from llama_index.llms.openai import OpenAI
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from autorag_research.pipelines.generation.basic_rag import BasicRAGPipeline
        from autorag_research.pipelines.retrieval.bm25 import BM25RetrievalPipeline

        engine = create_engine("postgresql://user:pass@localhost/dbname")
        session_factory = sessionmaker(bind=engine)

        # Create retrieval pipeline
        retrieval_pipeline = BM25RetrievalPipeline(
            session_factory=session_factory,
            name="bm25_baseline",
            index_path="/path/to/index",
        )

        # Create generation pipeline
        pipeline = BasicRAGPipeline(
            session_factory=session_factory,
            name="basic_rag_v1",
            llm=OpenAI(model="gpt-4"),
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
        llm: "LLM",
        retrieval_pipeline: "BaseRetrievalPipeline",
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        schema: Any | None = None,
    ):
        """Initialize Basic RAG pipeline.

        Args:
            session_factory: SQLAlchemy sessionmaker for database connections.
            name: Name for this pipeline.
            llm: LlamaIndex LLM instance for text generation.
            retrieval_pipeline: Retrieval pipeline for fetching relevant context.
            prompt_template: Template string with {context} and {query} placeholders.
            schema: Schema namespace from create_schema(). If None, uses default schema.
        """
        # Store prompt template before calling super().__init__
        # because _get_pipeline_config() is called in super().__init__
        self._prompt_template = prompt_template

        # Setup token counter for detailed token usage tracking
        self._token_counter = TokenCountingHandler()
        callback_manager = CallbackManager([self._token_counter])

        # Set callback manager on the LLM for token counting
        llm.callback_manager = callback_manager

        super().__init__(session_factory, name, llm, retrieval_pipeline, schema)

    def _get_pipeline_config(self) -> dict[str, Any]:
        """Return Basic RAG pipeline configuration."""
        return {
            "type": "basic_rag",
            "prompt_template": self._prompt_template,
            "retrieval_pipeline_id": self._retrieval_pipeline.pipeline_id,
        }

    def _get_chunk_contents(self, chunk_ids: list[int]) -> list[str]:
        """Get chunk contents by IDs.

        Args:
            chunk_ids: List of chunk IDs to fetch.

        Returns:
            List of chunk content strings in the same order as input IDs.
        """
        with GenerationUnitOfWork(self.session_factory, self._schema) as uow:
            chunks = uow.chunks.get_by_ids(chunk_ids)
            # Create a map for preserving order
            chunk_map = {chunk.id: chunk.contents for chunk in chunks}
            return [chunk_map.get(cid, "") for cid in chunk_ids]

    def _generate(self, query: str, top_k: int) -> GenerationResult:
        """Generate answer using simple RAG: retrieve once, generate once.

        Args:
            query: The query text to answer.
            top_k: Number of chunks to retrieve.

        Returns:
            GenerationResult containing the generated text and metadata.
        """
        # Reset token counter before each generation
        self._token_counter.reset_counts()

        # 1. Retrieve relevant chunks using composed retrieval pipeline
        retrieved = self._retrieval_pipeline.retrieve(query, top_k)

        # 2. Get chunk contents
        chunk_ids = [r["doc_id"] for r in retrieved]
        chunk_contents = self._get_chunk_contents(chunk_ids)

        # 3. Build prompt with context
        context = "\n\n".join(chunk_contents)
        prompt = self._prompt_template.format(context=context, query=query)

        # 4. Generate using LLM
        response = self._llm.complete(prompt)

        # Extract token usage from TokenCountingHandler
        token_usage = {
            "prompt_tokens": self._token_counter.prompt_llm_token_count,
            "completion_tokens": self._token_counter.completion_llm_token_count,
            "total_tokens": self._token_counter.total_llm_token_count,
            "embedding_tokens": self._token_counter.total_embedding_token_count,
        }

        # Fallback to response.raw for token usage if TokenCountingHandler didn't capture counts
        # (e.g., when LLM doesn't support callbacks or in mock scenarios)
        if token_usage["total_tokens"] == 0 and hasattr(response, "raw") and response.raw:
            usage = response.raw.get("usage", {})
            raw_prompt = usage.get("prompt_tokens", 0)
            raw_completion = usage.get("completion_tokens", 0)
            raw_total = usage.get("total_tokens", raw_prompt + raw_completion)
            if raw_total > 0:
                token_usage = {
                    "prompt_tokens": raw_prompt,
                    "completion_tokens": raw_completion,
                    "total_tokens": raw_total,
                    "embedding_tokens": 0,
                }

        return GenerationResult(
            text=str(response),
            token_usage=token_usage if token_usage["total_tokens"] > 0 else None,
            metadata={
                "retrieved_chunk_ids": chunk_ids,
                "retrieved_scores": [r["score"] for r in retrieved],
            },
        )
