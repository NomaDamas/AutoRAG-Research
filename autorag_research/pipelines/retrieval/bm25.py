"""BM25 Retrieval Pipeline for AutoRAG-Research.

This pipeline uses RetrievalPipelineService with BM25Module for retrieval.
"""

from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from autorag_research.orm.service.retrieval_pipeline import RetrievalPipelineService


class BM25RetrievalPipeline:
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
        results = pipeline.run(
            metric_id=1,
            top_k=10,
        )
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
        self.session_factory = session_factory
        self.name = name
        self.index_path = index_path
        self.k1 = k1
        self.b = b
        self.language = language
        self._schema = schema

        # Initialize service
        self._service = RetrievalPipelineService(session_factory, schema)

        # Create pipeline in DB
        self.pipeline_id = self._service.create_pipeline(
            name=name,
            config={
                "type": "bm25",
                "index_path": self.index_path,
                "k1": self.k1,
                "b": self.b,
                "language": self.language,
            },
        )

    def run(
        self,
        top_k: int = 10,
        batch_size: int = 100,
    ) -> dict[str, Any]:
        """Run BM25 retrieval pipeline.

        Args:
            top_k: Number of top documents to retrieve per query.
            batch_size: Number of queries to process in each batch.

        Returns:
            Dictionary with pipeline execution statistics:
            - pipeline_id: The pipeline ID
            - total_queries: Number of queries processed
            - total_results: Number of results stored
        """
        # Lazy import to avoid Java dependency at import time
        from autorag_research.nodes.retrieval.bm25 import BM25Module

        # Initialize BM25 module
        bm25 = BM25Module(
            index_path=self.index_path,
            k1=self.k1,
            b=self.b,
            language=self.language,
        )

        # Run pipeline with BM25 retrieval function
        return self._service.run_pipeline(
            retrieval_func=bm25.run,
            pipeline_id=self.pipeline_id,
            top_k=top_k,
            batch_size=batch_size,
        )
