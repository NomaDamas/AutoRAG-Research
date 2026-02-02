"""BM25 DB Retrieval Pipeline for AutoRAG-Research.

This pipeline uses VectorChord-BM25 for full-text BM25 retrieval
directly from database-stored chunks.
"""

from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from autorag_research.config import BaseRetrievalPipelineConfig
from autorag_research.pipelines.retrieval.base import BaseRetrievalPipeline

DEFAULT_BM25_INDEX_NAME = "idx_chunk_bm25"


@dataclass(kw_only=True)
class BM25PipelineConfig(BaseRetrievalPipelineConfig):
    """Configuration for BM25 retrieval pipeline.

    Attributes:
        name: Unique name for this pipeline instance.
        tokenizer: Tokenizer name for BM25 (default: "bert" for bert_base_uncased).
        index_name: Name of the BM25 index in PostgreSQL.
        top_k: Number of results to retrieve per query. Default: 10.
        batch_size: Number of queries to fetch from DB at once. Default: 128.
        max_concurrency: Maximum concurrent async operations. Default: 16.
        max_retries: Maximum retry attempts for failed queries. Default: 3.
        retry_delay: Base delay (seconds) for exponential backoff. Default: 1.0.

    Example:
        ```python
        config = BM25PipelineConfig(
            name="bm25_baseline",
            tokenizer="bert",
            top_k=10,
            max_concurrency=32,  # More parallelism for I/O-bound workloads
        )
        ```
    """

    tokenizer: str = "bert"
    """Tokenizer name for BM25 sparse retrieval.

    Available tokenizers (pg_tokenizer pre-built models):
        - "bert": bert-base-uncased (Hugging Face) - Default
        - "wiki_tocken": Wikitext-103 trained model
        - "gemma2b": Google lightweight model (~100MB memory)
        - "llmlingua2": Microsoft summarization model (~200MB memory)

    See: https://github.com/tensorchord/pg_tokenizer.rs/blob/main/docs/06-model.md
    """
    index_name: str = DEFAULT_BM25_INDEX_NAME

    def get_pipeline_class(self) -> type["BM25RetrievalPipeline"]:
        """Return the BM25RetrievalPipeline class."""
        return BM25RetrievalPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        """Return kwargs for BM25RetrievalPipeline constructor."""
        return {
            "tokenizer": self.tokenizer,
            "index_name": self.index_name,
        }


class BM25RetrievalPipeline(BaseRetrievalPipeline):
    """Pipeline for running VectorChord-BM25 retrieval.

    This pipeline wraps RetrievalPipelineService with BM25DBModule,
    providing a convenient interface for BM25-based retrieval using
    PostgreSQL's VectorChord-BM25 extension.

    BM25 does not require embeddings, so both _retrieve_by_id() and
    _retrieve_by_text() work without any embedding model.

    Example:
        ```python
        from autorag_research.orm.connection import DBConnection
        from autorag_research.pipelines.retrieval.bm25 import BM25RetrievalPipeline

        db = DBConnection.from_config()  # or DBConnection.from_env()
        session_factory = db.get_session_factory()

        # Initialize pipeline
        pipeline = BM25RetrievalPipeline(
            session_factory=session_factory,
            name="bm25_baseline",
            tokenizer="bert",
        )

        # Run pipeline on all queries in DB
        results = pipeline.run(top_k=10)

        # Or retrieve for a single query
        chunks = await pipeline.retrieve("What is machine learning?", top_k=10)
        ```
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        tokenizer: str = "bert",
        index_name: str = DEFAULT_BM25_INDEX_NAME,
        schema: Any | None = None,
    ):
        """Initialize BM25 retrieval pipeline.

        Args:
            session_factory: SQLAlchemy sessionmaker for database connections.
            name: Name for this pipeline.
            tokenizer: Tokenizer name for BM25 (default: "bert" for bert_base_uncased).
                Available tokenizers (pg_tokenizer pre-built models):
                    - "bert": bert-base-uncased (Hugging Face) - Default
                    - "wiki_tocken": Wikitext-103 trained model
                    - "gemma2b": Google lightweight model (~100MB memory)
                    - "llmlingua2": Microsoft summarization model (~200MB memory)
                See: https://github.com/tensorchord/pg_tokenizer.rs/blob/main/docs/06-model.md
            index_name: Name of the BM25 index (default: "idx_chunk_bm25").
            schema: Schema namespace from create_schema(). If None, uses default schema.
        """
        # Store BM25-specific parameters before calling super().__init__
        # because _get_pipeline_config() is called in super().__init__
        self.tokenizer = tokenizer
        self.index_name = index_name

        super().__init__(session_factory, name, schema)

    def _get_pipeline_config(self) -> dict[str, Any]:
        """Return BM25 pipeline configuration."""
        return {
            "type": "bm25",
            "tokenizer": self.tokenizer,
            "index_name": self.index_name,
        }

    async def _retrieve_by_id(self, query_id: int | str, top_k: int) -> list[dict[str, Any]]:
        """BM25 search using query ID.

        Args:
            query_id: The query ID to retrieve for.
            top_k: Number of top documents to retrieve.

        Returns:
            List of result dicts with doc_id, score, and content.
        """
        # Sync DB call (fast) - no need for true async
        results = self._service.bm25_search([query_id], top_k, tokenizer=self.tokenizer, index_name=self.index_name)
        return results[0] if results else []

    async def _retrieve_by_text(self, query_text: str, top_k: int) -> list[dict[str, Any]]:
        """BM25 search using raw text (no embedding needed).

        BM25 can search directly with text, no embedding computation required.

        Args:
            query_text: The query text to retrieve for.
            top_k: Number of top documents to retrieve.

        Returns:
            List of result dicts with doc_id, score, and content.
        """
        # BM25 can search directly with text - no embedding needed
        return self._service.bm25_search_by_text(
            query_text, top_k, tokenizer=self.tokenizer, index_name=self.index_name
        )
