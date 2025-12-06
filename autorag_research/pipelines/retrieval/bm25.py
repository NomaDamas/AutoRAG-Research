"""BM25 Retrieval Pipeline for AutoRAG-Research.

This pipeline uses RetrievalPipelineService with BM25Module for retrieval.
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
        index_path: Path to the Lucene index directory.
        k1: BM25 k1 parameter (controls term frequency saturation).
        b: BM25 b parameter (controls length normalization).
        language: Language for analyzer ('en', 'ko', 'zh', 'ja', etc.).
        top_k: Number of results to retrieve per query.
        batch_size: Number of queries to process in each batch.

    Example:
        ```python
        config = BM25PipelineConfig(
            name="bm25_baseline",
            index_path="/path/to/lucene/index",
            k1=0.9,
            b=0.4,
            top_k=10,
        )
        ```
    """

    index_path: str
    k1: float = 0.9
    b: float = 0.4
    language: str = "en"

    def get_pipeline_class(self) -> type["BM25RetrievalPipeline"]:
        """Return the BM25RetrievalPipeline class."""
        return BM25RetrievalPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        """Return kwargs for BM25RetrievalPipeline constructor."""
        return {
            "index_path": self.index_path,
            "k1": self.k1,
            "b": self.b,
            "language": self.language,
        }


class BM25RetrievalPipeline(BaseRetrievalPipeline):
    """Pipeline for running BM25 retrieval.

    This pipeline wraps RetrievalPipelineService with BM25Module,
    providing a convenient interface for BM25-based retrieval.

    Example:
        ```python
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from autorag_research.pipelines.retrieval.bm25 import BM25RetrievalPipeline

        engine = create_engine("postgresql://user:pass@localhost/dbname")
        session_factory = sessionmaker(bind=engine)

        # Initialize pipeline with index path
        pipeline = BM25RetrievalPipeline(
            session_factory=session_factory,
            name="bm25_baseline",
            index_path="/path/to/lucene/index",
        )

        # Run pipeline
        results = pipeline.run(top_k=10)
        ```
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        index_path: str,
        k1: float = 0.9,
        b: float = 0.4,
        language: str = "en",
        schema: Any | None = None,
    ):
        """Initialize BM25 retrieval pipeline.

        Args:
            session_factory: SQLAlchemy sessionmaker for database connections.
            name: Name for this pipeline.
            index_path: Path to the Lucene index directory.
            k1: BM25 k1 parameter (controls term frequency saturation).
            b: BM25 b parameter (controls length normalization).
            language: Language for analyzer ('en', 'ko', 'zh', 'ja', etc.).
            schema: Schema namespace from create_schema(). If None, uses default schema.
        """
        # Store BM25-specific parameters before calling super().__init__
        # because _get_pipeline_config() is called in super().__init__
        self.index_path = index_path
        self.k1 = k1
        self.b = b
        self.language = language

        super().__init__(session_factory, name, schema)

    def _get_pipeline_config(self) -> dict[str, Any]:
        """Return BM25 pipeline configuration."""
        return {
            "type": "bm25",
            "index_path": self.index_path,
            "k1": self.k1,
            "b": self.b,
            "language": self.language,
        }

    def _get_retrieval_func(self) -> Any:
        """Return BM25 retrieval function."""
        # Lazy import to avoid Java dependency at import time
        from autorag_research.nodes.retrieval.bm25 import BM25Module

        bm25 = BM25Module(
            index_path=self.index_path,
            k1=self.k1,
            b=self.b,
            language=self.language,
        )
        return bm25.run
