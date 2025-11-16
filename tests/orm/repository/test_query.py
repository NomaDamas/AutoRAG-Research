import pytest
from sqlalchemy.orm import Session

from autorag_research.orm.repository.query import QueryRepository


@pytest.fixture
def query_repository(db_session: Session) -> QueryRepository:
    return QueryRepository(db_session)


def test_get_by_query_text(query_repository: QueryRepository):
    result = query_repository.get_by_query_text("What is Doc One about?")

    assert result is not None
    assert result.query == "What is Doc One about?"
    assert result.generation_gt == ["alpha"]


def test_get_with_retrieval_relations(query_repository: QueryRepository):
    result = query_repository.get_with_retrieval_relations(1)

    assert result is not None
    assert hasattr(result, "retrieval_relations")


def test_get_with_experiment_results(query_repository: QueryRepository):
    result = query_repository.get_with_experiment_results(1)

    assert result is not None
    assert hasattr(result, "experiment_results")


def test_get_with_all_relations(query_repository: QueryRepository):
    result = query_repository.get_with_all_relations(1)

    assert result is not None
    assert hasattr(result, "retrieval_relations")
    assert hasattr(result, "experiment_results")
    assert hasattr(result, "chunk_retrieved_results")
    assert hasattr(result, "image_chunk_retrieved_results")


def test_search_by_query_text(query_repository: QueryRepository):
    results = query_repository.search_by_query_text("Doc", limit=10)

    assert len(results) >= 3
    assert all("doc" in q.query.lower() for q in results)


def test_get_queries_with_generation_gt(query_repository: QueryRepository):
    results = query_repository.get_queries_with_generation_gt()

    assert len(results) >= 5
    assert all(q.generation_gt is not None and len(q.generation_gt) > 0 for q in results)


def test_count_by_generation_gt_size(query_repository: QueryRepository):
    count = query_repository.count_by_generation_gt_size(1)

    assert count >= 5
