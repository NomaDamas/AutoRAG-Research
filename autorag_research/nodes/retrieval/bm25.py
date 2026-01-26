"""
BM25 retrieval module using VectorChord-BM25.

This module implements BM25 retrieval using PostgreSQL's VectorChord-BM25 extension,
enabling full-text search directly on database-stored chunks without external indices.
"""

from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from autorag_research.nodes import BaseModule
from autorag_research.orm.repository.chunk import ChunkRepository


class BM25Module(BaseModule):
    """
    BM25 retrieval module using VectorChord-BM25.

    This module performs BM25-based sparse retrieval using PostgreSQL's
    VectorChord-BM25 extension, querying chunks stored in the database.

    Attributes:
        session_factory: SQLAlchemy sessionmaker for database connections.
        tokenizer: Tokenizer name used for BM25 (default: "bert").
        index_name: Name of the BM25 index in PostgreSQL.
        schema: Schema namespace from create_schema().

    Example:
        ```python
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        engine = create_engine("postgresql://user:pass@localhost/dbname")
        session_factory = sessionmaker(bind=engine)

        module = BM25DBModule(session_factory=session_factory)
        results = module.run(["What is machine learning?"], top_k=10)
        ```
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        tokenizer: str = "bert",
        index_name: str = "idx_chunk_bm25",
        schema: Any | None = None,
    ):
        """
        Initialize BM25 DB module.

        Args:
            session_factory: SQLAlchemy sessionmaker for database connections.
            tokenizer: Tokenizer name for BM25 (default: "bert" for bert_base_uncased).
                       Other options may include: "simple", "standard", etc.
            index_name: Name of the BM25 index (default: "idx_chunk_bm25").
            schema: Schema namespace from create_schema(). If None, uses default schema.
        """
        self.session_factory = session_factory
        self.tokenizer = tokenizer
        self.index_name = index_name
        self._schema = schema

    def _get_chunk_model(self) -> type:
        """Get the Chunk model class from schema or default."""
        if self._schema is not None:
            return self._schema.Chunk
        from autorag_research.orm.schema import Chunk

        return Chunk

    def run(self, queries: list[str], top_k: int = 10) -> list[list[dict]]:
        """
        Execute BM25 retrieval for given queries.

        Args:
            queries: List of query strings for batch processing.
            top_k: Number of top documents to retrieve per query.

        Returns:
            List of search results for each query. Each query returns a list of top_k results.

            Each result dictionary contains:
            - doc_id: Chunk ID (int)
            - score: BM25 relevance score (higher = more relevant)
            - content: Chunk text content
        """
        all_results = []
        chunk_model = self._get_chunk_model()

        with self.session_factory() as session:
            repo = ChunkRepository(session, chunk_model)

            for query in queries:
                # Perform BM25 search
                results = repo.bm25_search(
                    query_text=query,
                    index_name=self.index_name,
                    limit=top_k,
                    tokenizer=self.tokenizer,
                )

                query_results = []
                for chunk, score in results:
                    result = {
                        "doc_id": chunk.id,
                        "score": score,
                        "content": chunk.contents,
                    }
                    query_results.append(result)

                all_results.append(query_results)

        return all_results
