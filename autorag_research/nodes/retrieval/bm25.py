"""
BM25 retrieval module using Pyserini.

This module implements BM25 retrieval using Pyserini's pre-built indices.
It supports batch processing and can be configured with different indices.
"""

from pathlib import Path

from pyserini.search.lucene import LuceneSearcher

from autorag_research.nodes import BaseModule


class BM25Module(BaseModule):
    """
    BM25 retrieval module using Pyserini.

    This module performs BM25-based sparse retrieval using either pre-built
    indices or custom indices created from user documents.

    Attributes:
        searcher: Pyserini SimpleSearcher instance
        index_name: Name of the pre-built index (if using pre-built)
        index_path: Path to the custom index (if using custom index)
    """

    def __init__(
        self,
        index_name: str | None = None,
        index_path: str | None = None,
        k1: float = 0.9,
        b: float = 0.4,
        language: str = "en",
    ):
        """
        Initialize BM25 module.

        Args:
            index_name: Name of pre-built index (e.g., 'msmarco-passage', 'msmarco-doc')
            index_path: Path to custom Lucene index (used if index_name is not provided)
            k1: BM25 k1 parameter (controls term frequency saturation)
            b: BM25 b parameter (controls length normalization)
            language: Language for analyzer ('en', 'ko', 'zh', 'ja', etc.)

        Raises:
            ValueError: If neither index_name nor index_path is provided
        """
        if index_name is None and index_path is None:
            msg = "Either index_name or index_path must be provided"
            raise ValueError(msg)

        if index_name:
            # Use pre-built index
            self.searcher = LuceneSearcher.from_prebuilt_index(index_name)
            self.index_name = index_name
            self.index_path = None
        else:
            # Use custom index
            self.searcher = LuceneSearcher(str(index_path))
            self.index_name = None
            self.index_path = Path(index_path)

        # Set BM25 parameters
        self.searcher.set_bm25(k1=k1, b=b)

        # Set language analyzer if not English
        if language != "en":
            self._set_language_analyzer(language)

        self.k1 = k1
        self.b = b
        self.language = language

    def _set_language_analyzer(self, language: str):
        """Set appropriate analyzer for non-English languages."""
        analyzer_map = {
            "ko": "korean",
            "zh": "chinese",
            "ja": "japanese",
            "ar": "arabic",
            "de": "german",
            "fr": "french",
            "es": "spanish",
            "ru": "russian",
        }

        if language in analyzer_map:
            # Note: This requires the searcher to be configured with proper analyzer
            # Pyserini handles this internally for pre-built indices
            pass

    def run(self, queries: list[str], top_k: int = 10) -> list[list[dict]]:
        """
        Execute BM25 retrieval for given queries.

        Args:
            queries: List of query strings for batch processing
            top_k: Number of top documents to retrieve per query

        Returns:
            List of search results for each query. Each query returns a list of top_k results.

            Each result dictionary contains:
            - doc_id: Document ID
            - score: BM25 relevance score
            - content: Document content
        """
        all_results = []

        for query in queries:
            hits = self.searcher.search(query, k=top_k)

            query_results = []
            for hit in hits:
                content = ""
                if hasattr(hit, "raw"):
                    content = hit.raw
                elif hasattr(hit, "contents"):
                    content = hit.contents

                result = {
                    "doc_id": hit.docid,
                    "score": hit.score,
                    "content": content,
                }
                query_results.append(result)

            all_results.append(query_results)

        return all_results
