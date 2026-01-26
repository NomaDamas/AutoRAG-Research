"""BM25 DB Retrieval Pipeline for AutoRAG-Research.

This pipeline uses VectorChord-BM25 for full-text BM25 retrieval
directly from database-stored chunks.
"""

from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from autorag_research.config import BaseRetrievalPipelineConfig
from autorag_research.pipelines.retrieval.base import BaseRetrievalPipeline


@dataclass(kw_only=True)
class BM25PipelineConfig(BaseRetrievalPipelineConfig):
    """Configuration for BM25 retrieval pipeline.

    Attributes:
        name: Unique name for this pipeline instance.
        tokenizer: Tokenizer name for BM25 (default: "bert" for bert_base_uncased).
        index_name: Name of the BM25 index in PostgreSQL.
        top_k: Number of results to retrieve per query.
        batch_size: Number of queries to process in each batch.

    Example:
        ```python
        config = BM25PipelineConfig(
            name="bm25_baseline",
            tokenizer="bert",
            top_k=10,
        )
        ```
    """

    tokenizer: str = "bert"
    index_name: str = "idx_chunk_bm25"

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

    Example:
        ```python
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from autorag_research.pipelines.retrieval.bm25 import BM25RetrievalPipeline

        engine = create_engine("postgresql://user:pass@localhost/dbname")
        session_factory = sessionmaker(bind=engine)

        # Initialize pipeline
        pipeline = BM25RetrievalPipeline(
            session_factory=session_factory,
            name="bm25_baseline",
            tokenizer="bert",
        )

        # Run pipeline on all queries in DB
        results = pipeline.run(top_k=10)

        # Or retrieve for a single query
        chunks = pipeline.retrieve("What is machine learning?", top_k=10)
        ```
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        tokenizer: str = "bert",
        index_name: str = "idx_chunk_bm25",
        schema: Any | None = None,
    ):
        """Initialize BM25 retrieval pipeline.

        Args:
            session_factory: SQLAlchemy sessionmaker for database connections.
            name: Name for this pipeline.
            tokenizer: Tokenizer name for BM25 (default: "bert" for bert_base_uncased).
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

    def _get_retrieval_func(self) -> Any:
        """Return BM25 retrieval function."""
        from autorag_research.nodes.retrieval.bm25 import BM25DBModule

        bm25 = BM25DBModule(
            session_factory=self.session_factory,
            tokenizer=self.tokenizer,
            index_name=self.index_name,
            schema=self._schema,
        )
        return bm25.run
