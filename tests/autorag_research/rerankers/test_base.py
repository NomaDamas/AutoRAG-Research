"""Tests for base reranker module."""

import pytest

from autorag_research.rerankers.base import BaseReranker, RerankResult


class MockReranker(BaseReranker):
    """Mock reranker for testing base class functionality."""

    def rerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        """Mock rerank that scores based on word overlap."""
        if not documents:
            return []

        top_k = top_k or len(documents)
        query_words = set(query.lower().split())

        results = []
        for i, doc in enumerate(documents):
            doc_words = set(doc.lower().split())
            score = len(query_words & doc_words) / max(len(query_words | doc_words), 1)
            results.append(RerankResult(index=i, text=doc, score=score))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    async def arerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        """Async mock rerank."""
        return self.rerank(query, documents, top_k)


class TestRerankResult:
    """Tests for RerankResult dataclass."""

    def test_create_rerank_result(self) -> None:
        """Can create RerankResult with all fields."""
        result = RerankResult(index=0, text="test document", score=0.95)

        assert result.index == 0
        assert result.text == "test document"
        assert result.score == 0.95

    def test_rerank_result_equality(self) -> None:
        """RerankResult instances with same values are equal."""
        result1 = RerankResult(index=0, text="test", score=0.5)
        result2 = RerankResult(index=0, text="test", score=0.5)

        assert result1 == result2


class TestBaseReranker:
    """Tests for BaseReranker abstract class."""

    def test_default_model_name(self) -> None:
        """Default model_name is 'unknown'."""
        reranker = MockReranker()
        assert reranker.model_name == "unknown"

    def test_custom_model_name(self) -> None:
        """Can set custom model_name."""
        reranker = MockReranker(model_name="test-model")
        assert reranker.model_name == "test-model"

    def test_default_batch_size(self) -> None:
        """Default batch_size is 64."""
        reranker = MockReranker()
        assert reranker.batch_size == 64

    def test_custom_batch_size(self) -> None:
        """Can set custom batch_size."""
        reranker = MockReranker(batch_size=32)
        assert reranker.batch_size == 32


class TestRerank:
    """Tests for rerank method."""

    @pytest.fixture
    def reranker(self) -> MockReranker:
        """Create mock reranker instance."""
        return MockReranker()

    def test_rerank_empty_documents(self, reranker: MockReranker) -> None:
        """Rerank returns empty list for empty documents."""
        results = reranker.rerank("query", [])
        assert results == []

    def test_rerank_single_document(self, reranker: MockReranker) -> None:
        """Rerank works with single document."""
        results = reranker.rerank("test query", ["test document"])

        assert len(results) == 1
        assert results[0].index == 0
        assert results[0].text == "test document"

    def test_rerank_multiple_documents(self, reranker: MockReranker) -> None:
        """Rerank orders documents by relevance."""
        documents = [
            "unrelated content about cats",
            "test query document",
            "another unrelated text",
        ]
        results = reranker.rerank("test query", documents)

        assert len(results) == 3
        # Document with "test query" should be ranked first
        assert results[0].index == 1
        assert results[0].text == documents[1]

    def test_rerank_top_k(self, reranker: MockReranker) -> None:
        """Rerank respects top_k parameter."""
        documents = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        results = reranker.rerank("query", documents, top_k=2)

        assert len(results) == 2

    def test_rerank_scores_descending(self, reranker: MockReranker) -> None:
        """Rerank returns results in descending score order."""
        documents = ["low relevance", "test query match", "medium relevance test"]
        results = reranker.rerank("test query", documents)

        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score


class TestArerank:
    """Tests for async arerank method."""

    @pytest.fixture
    def reranker(self) -> MockReranker:
        """Create mock reranker instance."""
        return MockReranker()

    @pytest.mark.asyncio
    async def test_arerank_empty_documents(self, reranker: MockReranker) -> None:
        """Async rerank returns empty list for empty documents."""
        results = await reranker.arerank("query", [])
        assert results == []

    @pytest.mark.asyncio
    async def test_arerank_single_document(self, reranker: MockReranker) -> None:
        """Async rerank works with single document."""
        results = await reranker.arerank("test query", ["test document"])

        assert len(results) == 1
        assert results[0].index == 0

    @pytest.mark.asyncio
    async def test_arerank_top_k(self, reranker: MockReranker) -> None:
        """Async rerank respects top_k parameter."""
        documents = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        results = await reranker.arerank("query", documents, top_k=3)

        assert len(results) == 3


class TestRerankDocuments:
    """Tests for rerank_documents batch method."""

    @pytest.fixture
    def reranker(self) -> MockReranker:
        """Create mock reranker instance."""
        return MockReranker()

    def test_rerank_documents_empty(self, reranker: MockReranker) -> None:
        """Rerank documents returns empty for empty inputs."""
        results = reranker.rerank_documents([], [])
        assert results == []

    def test_rerank_documents_single_query(self, reranker: MockReranker) -> None:
        """Rerank documents works with single query."""
        queries = ["test query"]
        documents_list = [["doc1", "doc2"]]

        results = reranker.rerank_documents(queries, documents_list)

        assert len(results) == 1
        assert len(results[0]) == 2

    def test_rerank_documents_multiple_queries(self, reranker: MockReranker) -> None:
        """Rerank documents works with multiple queries."""
        queries = ["query1", "query2", "query3"]
        documents_list = [["doc1a", "doc1b"], ["doc2a", "doc2b"], ["doc3a", "doc3b"]]

        results = reranker.rerank_documents(queries, documents_list)

        assert len(results) == 3
        for result in results:
            assert len(result) == 2

    def test_rerank_documents_with_top_k(self, reranker: MockReranker) -> None:
        """Rerank documents respects top_k."""
        queries = ["q1", "q2"]
        documents_list = [["d1", "d2", "d3"], ["d4", "d5", "d6"]]

        results = reranker.rerank_documents(queries, documents_list, top_k=1)

        assert len(results) == 2
        assert len(results[0]) == 1
        assert len(results[1]) == 1


class TestArankDocuments:
    """Tests for async arerank_documents batch method."""

    @pytest.fixture
    def reranker(self) -> MockReranker:
        """Create mock reranker instance."""
        return MockReranker()

    @pytest.mark.asyncio
    async def test_arerank_documents_multiple_queries(self, reranker: MockReranker) -> None:
        """Async rerank documents works with multiple queries."""
        queries = ["query1", "query2"]
        documents_list = [["doc1", "doc2"], ["doc3", "doc4"]]

        results = await reranker.arerank_documents(queries, documents_list)

        assert len(results) == 2
        assert len(results[0]) == 2
        assert len(results[1]) == 2


class TestRerankDocumentsBatch:
    """Tests for batched rerank_documents_batch method."""

    @pytest.fixture
    def reranker(self) -> MockReranker:
        """Create mock reranker with small batch size."""
        return MockReranker(batch_size=2)

    def test_rerank_documents_batch_small_input(self, reranker: MockReranker) -> None:
        """Batched rerank works when input fits in single batch."""
        queries = ["q1"]
        documents_list = [["d1", "d2"]]

        results = reranker.rerank_documents_batch(queries, documents_list)

        assert len(results) == 1

    def test_rerank_documents_batch_large_input(self, reranker: MockReranker) -> None:
        """Batched rerank splits input into batches."""
        queries = ["q1", "q2", "q3", "q4", "q5"]
        documents_list = [["d1"], ["d2"], ["d3"], ["d4"], ["d5"]]

        results = reranker.rerank_documents_batch(queries, documents_list)

        assert len(results) == 5


class TestArankDocumentsBatch:
    """Tests for async batched arerank_documents_batch method."""

    @pytest.fixture
    def reranker(self) -> MockReranker:
        """Create mock reranker with small batch size."""
        return MockReranker(batch_size=2)

    @pytest.mark.asyncio
    async def test_arerank_documents_batch_large_input(self, reranker: MockReranker) -> None:
        """Async batched rerank works with large input."""
        queries = ["q1", "q2", "q3", "q4"]
        documents_list = [["d1"], ["d2"], ["d3"], ["d4"]]

        results = await reranker.arerank_documents_batch(queries, documents_list)

        assert len(results) == 4
