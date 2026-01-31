"""HyDE (Hypothetical Document Embeddings) Retrieval Pipeline for AutoRAG-Research.

This pipeline implements the HyDE retrieval approach from the paper
"Precise Zero-Shot Dense Retrieval without Relevance Labels" (Gao et al., 2022).

HyDE works by:
1. Using an LLM to generate a hypothetical document that would answer the query
2. Embedding the hypothetical document (not the original query)
3. Performing vector similarity search with the hypothetical embedding

This bridges the semantic gap between queries and documents.
"""

from dataclasses import dataclass, field
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.config import BaseRetrievalPipelineConfig
from autorag_research.injection import health_check_embedding, health_check_llm
from autorag_research.pipelines.retrieval.base import BaseRetrievalPipeline

DEFAULT_HYDE_PROMPT_TEMPLATE = """Please write a passage to answer the question.
Question: {question}
Passage:"""


@dataclass(kw_only=True)
class HyDEPipelineConfig(BaseRetrievalPipelineConfig):
    """Configuration for HyDE retrieval pipeline.

    Attributes:
        name: Unique name for this pipeline instance.
        llm: LLM config name or instance for generating hypothetical documents.
        embedding: Embedding config name or instance for embedding the hypothetical doc.
        prompt_template: Template with {question} placeholder for generating hypothetical docs.
        top_k: Number of results to retrieve per query.
        batch_size: Number of queries to process in each batch.

    Example:
        ```python
        config = HyDEPipelineConfig(
            name="hyde_gpt4",
            llm="openai-gpt4",
            embedding="openai-small",
            top_k=10,
        )
        ```
    """

    llm: str | BaseLanguageModel
    """LLM for generating hypothetical documents. Can be config name or instance."""

    embedding: str | Embeddings
    """Embedding model for the hypothetical document. Can be config name or instance."""

    prompt_template: str = field(default=DEFAULT_HYDE_PROMPT_TEMPLATE)
    """Template with {question} placeholder for generating hypothetical documents."""

    def __setattr__(self, name: str, value: Any) -> None:
        """Auto-convert string config names to model instances."""
        if name == "llm" and isinstance(value, str):
            from autorag_research.injection import load_llm

            value = load_llm(value)
            health_check_llm(value)
        elif name == "embedding" and isinstance(value, str):
            from autorag_research.injection import load_embedding_model

            value = load_embedding_model(value)
            health_check_embedding(value)
        super().__setattr__(name, value)

    def get_pipeline_class(self) -> type["HyDERetrievalPipeline"]:
        """Return the HyDERetrievalPipeline class."""
        return HyDERetrievalPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        """Return kwargs for HyDERetrievalPipeline constructor."""
        return {
            "llm": self.llm,
            "embedding": self.embedding,
            "prompt_template": self.prompt_template,
        }


class HyDERetrievalPipeline(BaseRetrievalPipeline):
    """Pipeline for HyDE (Hypothetical Document Embeddings) retrieval.

    This pipeline generates hypothetical documents using an LLM, then embeds
    those documents for vector search. This approach bridges the semantic gap
    between short queries and longer documents.

    Reference: "Precise Zero-Shot Dense Retrieval without Relevance Labels"
    https://arxiv.org/abs/2212.10496

    Example:
        ```python
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from autorag_research.orm.connection import DBConnection
        from autorag_research.pipelines.retrieval.hyde import HyDERetrievalPipeline

        db = DBConnection.from_config()
        session_factory = db.get_session_factory()

        pipeline = HyDERetrievalPipeline(
            session_factory=session_factory,
            name="hyde_gpt4",
            llm=ChatOpenAI(model="gpt-4"),
            embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        )

        # Single query retrieval
        results = pipeline.retrieve("What is machine learning?", top_k=10)

        # Batch processing
        stats = pipeline.run(top_k=10)
        ```
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        llm: BaseLanguageModel,
        embedding: Embeddings,
        prompt_template: str = DEFAULT_HYDE_PROMPT_TEMPLATE,
        schema: Any | None = None,
    ):
        """Initialize HyDE retrieval pipeline.

        Args:
            session_factory: SQLAlchemy sessionmaker for database connections.
            name: Name for this pipeline.
            llm: LangChain LLM for generating hypothetical documents.
            embedding: LangChain embeddings model for embedding hypothetical docs.
            prompt_template: Template with {question} placeholder for generating
                hypothetical documents. Defaults to the paper's general template.
            schema: Schema namespace from create_schema(). If None, uses default schema.
        """
        # Store parameters BEFORE calling super().__init__
        # because _get_pipeline_config() is called in super().__init__
        self.llm = llm
        self.embedding = embedding
        self.prompt_template = prompt_template

        super().__init__(session_factory, name, schema)

    def _get_pipeline_config(self) -> dict[str, Any]:
        """Return HyDE pipeline configuration.

        Returns:
            Dictionary containing pipeline configuration for storage.
        """
        return {
            "type": "hyde",
            "prompt_template": self.prompt_template,
        }

    def _generate_hypothetical_document(self, query_text: str) -> str:
        """Generate a hypothetical document using the LLM.

        Args:
            query_text: The query to generate a hypothetical document for.

        Returns:
            Generated hypothetical document text.
        """
        prompt = self.prompt_template.format(question=query_text)
        response = self.llm.invoke(prompt)
        # Handle both string and AIMessage responses
        if hasattr(response, "content"):
            return str(response.content)
        return str(response)

    def _get_retrieval_func(self) -> Any:
        """Return HyDE retrieval function.

        Returns:
            A callable that generates hypothetical documents, embeds them,
            and performs vector search.
        """

        def hyde_retrieval(query_ids: list[int], top_k: int) -> list[list[dict[str, Any]]]:
            all_results: list[list[dict[str, Any]]] = []
            Query = self._get_query_model()

            for query_id in query_ids:
                # Get query text from DB
                with self.session_factory() as session:
                    query = session.get(Query, query_id)
                    if query is None:
                        raise ValueError(f"Query {query_id} not found")  # noqa: TRY003
                    query_text = query.contents  # type: ignore[union-attr]

                # Generate hypothetical document
                hypothetical_doc = self._generate_hypothetical_document(query_text)

                # Embed the hypothetical document
                hypothetical_embedding = self.embedding.embed_query(hypothetical_doc)

                # Search using the hypothetical embedding
                results = self._service.vector_search_by_embedding(
                    embedding=hypothetical_embedding,
                    top_k=top_k,
                )
                all_results.append(results)

            return all_results

        return hyde_retrieval


__all__ = ["DEFAULT_HYDE_PROMPT_TEMPLATE", "HyDEPipelineConfig", "HyDERetrievalPipeline"]
