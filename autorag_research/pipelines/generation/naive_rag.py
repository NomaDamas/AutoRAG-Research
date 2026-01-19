"""Naive RAG Pipeline for AutoRAG-Research.

Implements simple single-call RAG: retrieve once -> build prompt -> generate once.
"""

from dataclasses import dataclass, field
from typing import Any

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
class NaiveRAGPipelineConfig(BaseGenerationPipelineConfig):
    """Configuration for Naive RAG pipeline.

    Attributes:
        name: Unique name for this pipeline instance.
        llm: LlamaIndex LLM instance for text generation.
        retrieval_pipeline: Retrieval pipeline for context retrieval.
        prompt_template: Template for building the generation prompt.
            Must contain {context} and {query} placeholders.
        top_k: Number of chunks to retrieve per query.
        batch_size: Number of queries to process in each batch.

    Example:
        ```python
        from llama_index.llms.openai import OpenAI

        config = NaiveRAGPipelineConfig(
            name="naive_rag_v1",
            llm=OpenAI(model="gpt-4"),
            retrieval_pipeline=my_retrieval_pipeline,
            prompt_template="Context:\\n{context}\\n\\nQ: {query}\\nA:",
            top_k=5,
        )
        ```
    """

    llm: "LLM"
    retrieval_pipeline: "BaseRetrievalPipeline"
    prompt_template: str = field(default=DEFAULT_PROMPT_TEMPLATE)

    def get_pipeline_class(self) -> type["NaiveRAGPipeline"]:
        """Return the NaiveRAGPipeline class."""
        return NaiveRAGPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        """Return kwargs for NaiveRAGPipeline constructor."""
        return {
            "llm": self.llm,
            "retrieval_pipeline": self.retrieval_pipeline,
            "prompt_template": self.prompt_template,
        }

    def get_run_kwargs(self) -> dict[str, Any]:
        """Return kwargs for pipeline.run() method."""
        return {"top_k": self.top_k, "batch_size": self.batch_size}


class NaiveRAGPipeline(BaseGenerationPipeline):
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

        from autorag_research.pipelines.generation.naive_rag import NaiveRAGPipeline
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
        pipeline = NaiveRAGPipeline(
            session_factory=session_factory,
            name="naive_rag_v1",
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
        """Initialize Naive RAG pipeline.

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

        super().__init__(session_factory, name, llm, retrieval_pipeline, schema)

    def _get_pipeline_config(self) -> dict[str, Any]:
        """Return Naive RAG pipeline configuration."""
        return {
            "type": "naive_rag",
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

        # Extract token usage if available
        token_usage = None
        if hasattr(response, "raw") and response.raw:
            usage = response.raw.get("usage", {})
            token_usage = usage.get("total_tokens")

        return GenerationResult(
            text=str(response),
            token_usage=token_usage,
            metadata={
                "retrieved_chunk_ids": chunk_ids,
                "retrieved_scores": [r["score"] for r in retrieved],
            },
        )
