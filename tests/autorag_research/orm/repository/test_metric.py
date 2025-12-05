import pytest
from sqlalchemy.orm import Session

from autorag_research.orm.repository.metric import MetricRepository


@pytest.fixture
def metric_repository(db_session: Session) -> MetricRepository:
    return MetricRepository(db_session)


def test_get_by_name(metric_repository: MetricRepository):
    result = metric_repository.get_by_name("retrieval@k")

    assert result is not None
    assert result.name == "retrieval@k"
    assert result.type == "retrieval"


def test_get_by_name_and_type(metric_repository: MetricRepository):
    result = metric_repository.get_by_name_and_type("bleu", "generation")

    assert result is not None
    assert result.name == "bleu"
    assert result.type == "generation"


def test_get_by_type(metric_repository: MetricRepository):
    results = metric_repository.get_by_type("retrieval")

    assert len(results) >= 1
    assert all(m.type == "retrieval" for m in results)


def test_get_with_summaries(metric_repository: MetricRepository):
    result = metric_repository.get_with_summaries(1)

    assert result is not None
    assert hasattr(result, "summaries")


def test_get_with_all_relations(metric_repository: MetricRepository):
    result = metric_repository.get_with_all_relations(1)

    assert result is not None
    assert hasattr(result, "summaries")


def test_search_by_name(metric_repository: MetricRepository):
    results = metric_repository.search_by_name("retrieval", limit=10)

    assert len(results) >= 1
    assert any("retrieval" in m.name.lower() for m in results)


def test_get_all_retrieval_metrics(metric_repository: MetricRepository):
    results = metric_repository.get_all_retrieval_metrics()

    assert len(results) >= 1
    assert all(m.type == "retrieval" for m in results)


def test_get_all_generation_metrics(metric_repository: MetricRepository):
    results = metric_repository.get_all_generation_metrics()

    assert len(results) >= 1
    assert all(m.type == "generation" for m in results)


def test_exists_by_name(metric_repository: MetricRepository):
    exists = metric_repository.exists_by_name("retrieval@k")
    assert exists is True

    not_exists = metric_repository.exists_by_name("nonexistent_metric")
    assert not_exists is False


def test_exists_by_name_and_type(metric_repository: MetricRepository):
    exists = metric_repository.exists_by_name_and_type("bleu", "generation")
    assert exists is True

    not_exists = metric_repository.exists_by_name_and_type("bleu", "retrieval")
    assert not_exists is False
