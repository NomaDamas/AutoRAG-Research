"""
BM25 retrieval module using VectorChord-BM25.

This module implements BM25 retrieval using PostgreSQL's VectorChord-BM25 extension,
enabling full-text search directly on database-stored chunks without external indices.
"""

from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from autorag_research.nodes import BaseModule, make_retrieval_result
from autorag_research.orm.repository.chunk import ChunkRepository
from autorag_research.orm.repository.query import QueryRepository


class BM25Module(BaseModule):
    """
    BM25 retrieval module using VectorChord-BM25.

    This module performs BM25-based sparse retrieval using PostgreSQL's
    VectorChord-BM25 extension, querying chunks stored in the database.

    Attributes:
        session_factory: SQLAlchemy sessionmaker for database connections.
        tokenizer: Tokenizer name used for BM25 (default: "bert").
            For available models, see: https://github.com/tensorchord/pg_tokenizer.rs/blob/main/docs/06-model.md
        index_name: Name of the BM25 index in PostgreSQL.
        schema: Schema namespace from create_schema().

    Example:
        ```python
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        engine = create_engine("postgresql://user:pass@localhost/dbname")
        session_factory = sessionmaker(bind=engine)

        module = BM25Module(session_factory=session_factory)
        results = module.run([1, 2, 3], top_k=10)  # Query IDs
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
                Available tokenizers (pg_tokenizer pre-built models):
                    - "bert": bert-base-uncased (Hugging Face) - Default
                    - "wiki_tocken": Wikitext-103 trained model
                    - "gemma2b": Google lightweight model (~100MB memory)
                    - "llmlingua2": Microsoft summarization model (~200MB memory)
                See: https://github.com/tensorchord/pg_tokenizer.rs/blob/main/docs/06-model.md
            index_name: Name of the BM25 index (default: "idx_chunk_bm25").
            schema: Schema namespace from create_schema(). If None, uses default schema.
        """
        self.session_factory = session_factory
        self.tokenizer = tokenizer
        self.index_name = index_name
        self._schema = schema

    def search(self, query_texts: list[str], top_k: int = 10) -> list[list[dict]]:
        """
        Execute BM25 retrieval with provided texts directly.

        Bypasses Query table lookup - use for query expansion, rewriting, etc.

        Args:
            query_texts: Texts to search with.
            top_k: Number of results per query.

        Returns:
            List of search results for each query. Each query returns a list of top_k results.

            Each result dictionary contains:
            - doc_id: Chunk ID (int)
            - score: BM25 relevance score (higher = more relevant)
            - content: Chunk text content

        Example:
            ```python
            # Direct text search (e.g., query expansion)
            expanded_queries = ["machine learning basics", "ML fundamentals"]
            results = module.search(expanded_queries, top_k=10)

            # Rewritten query search
            rewritten = llm.complete("Rewrite: What is ML?").text
            results = module.search([rewritten], top_k=10)
            ```
        """
        if not query_texts:
            return []

        all_results = []
        chunk_model = self._get_chunk_model()

        with self.session_factory() as session:
            chunk_repo = ChunkRepository(session, chunk_model)

            for query_text in query_texts:
                results = chunk_repo.bm25_search(
                    query_text=query_text,
                    index_name=self.index_name,
                    limit=top_k,
                    tokenizer=self.tokenizer,
                )
                all_results.append([make_retrieval_result(chunk, score) for chunk, score in results])

        return all_results

    def run(self, query_ids: list[int | str], top_k: int = 10) -> list[list[dict]]:
        """
        Execute BM25 retrieval for given query IDs.

        Args:
            query_ids: List of query IDs to search.
            top_k: Number of top documents to retrieve per query.

        Returns:
            List of search results for each query. Each query returns a list of top_k results.

            Each result dictionary contains:
            - doc_id: Chunk ID (int)
            - score: BM25 relevance score (higher = more relevant)
            - content: Chunk text content

        Raises:
            ValueError: If a query is not found.
        """
        all_results = []
        chunk_model = self._get_chunk_model()
        query_model = self._get_query_model()

        with self.session_factory() as session:
            chunk_repo = ChunkRepository(session, chunk_model)
            query_repo = QueryRepository(session, query_model)

            for query_id in query_ids:
                # Look up query to get text
                query = query_repo.get_by_id(query_id)
                if query is None:
                    raise ValueError(f"Query {query_id} not found")  # noqa: TRY003

                # Perform BM25 search using query text
                results = chunk_repo.bm25_search(
                    query_text=query.contents,
                    index_name=self.index_name,
                    limit=top_k,
                    tokenizer=self.tokenizer,
                )
                all_results.append([make_retrieval_result(chunk, score) for chunk, score in results])

        return all_results
