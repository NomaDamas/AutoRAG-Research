import pytest
from sqlalchemy.orm import Session

from autorag_research.orm.repository.query import QueryRepository


@pytest.fixture
def query_repository(db_session: Session) -> QueryRepository:
    return QueryRepository(db_session)


def test_get_by_query_text(query_repository: QueryRepository):
    result = query_repository.get_by_query_text("What is Doc One about?")

    assert result is not None
    assert result.contents == "What is Doc One about?"
    assert result.generation_gt == ["alpha"]


def test_get_with_retrieval_relations(query_repository: QueryRepository):
    result = query_repository.get_with_retrieval_relations(1)

    assert result is not None
    assert hasattr(result, "query_to_llm")
    assert hasattr(result, "retrieval_relations")


def test_get_with_executor_results(query_repository: QueryRepository):
    result = query_repository.get_with_executor_results(1)

    assert result is not None
    assert hasattr(result, "executor_results")


def test_get_with_all_relations(query_repository: QueryRepository):
    result = query_repository.get_with_all_relations(1)

    assert result is not None
    assert hasattr(result, "retrieval_relations")
    assert hasattr(result, "executor_results")
    assert hasattr(result, "chunk_retrieved_results")
    assert hasattr(result, "image_chunk_retrieved_results")
    assert hasattr(result, "evaluation_results")


def test_search_by_query_text(query_repository: QueryRepository):
    results = query_repository.search_by_query_text("Doc", limit=10)

    assert len(results) >= 3
    assert all("doc" in q.contents.lower() for q in results)


def test_get_queries_with_generation_gt(query_repository: QueryRepository):
    results = query_repository.get_queries_with_generation_gt()

    assert len(results) >= 5
    assert all(q.generation_gt is not None and len(q.generation_gt) > 0 for q in results)


def test_count_by_generation_gt_size(query_repository: QueryRepository):
    count = query_repository.count_by_generation_gt_size(1)

    assert count >= 5


def test_query_has_embedding_attributes(query_repository: QueryRepository):
    """Test that Query entities have embedding and embeddings attributes."""
    result = query_repository.get_by_id(1)

    assert result is not None
    assert hasattr(result, "embedding")
    assert hasattr(result, "embeddings")
    # Verify they can be None (as per seed data)
    assert result.embedding is None
    assert result.embeddings is None


def test_vector_search_capability(query_repository: QueryRepository):
    assert hasattr(query_repository, "vector_search")
    assert hasattr(query_repository, "vector_search_with_scores")


def test_find_by_contents(query_repository: QueryRepository):
    result = query_repository.find_by_contents("What is Doc One about?")

    assert result is not None
    assert result.contents == "What is Doc One about?"
    assert result.id == 1
