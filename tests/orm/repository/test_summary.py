import pytest
from sqlalchemy.orm import Session

from autorag_research.orm.repository.summary import SummaryRepository
from autorag_research.orm.schema import Summary


@pytest.fixture
def summary_repository(db_session: Session) -> SummaryRepository:
    return SummaryRepository(db_session)


def test_get_by_composite_key(summary_repository: SummaryRepository):
    result = summary_repository.get_by_composite_key(1, 1)

    assert result is not None
    assert result.pipeline_id == 1
    assert result.metric_id == 1
    assert result.metric_result == 0.82


def test_get_by_pipeline_id(summary_repository: SummaryRepository):
    results = summary_repository.get_by_pipeline_id(1)

    assert len(results) >= 1
    assert all(s.pipeline_id == 1 for s in results)


def test_get_by_metric_id(summary_repository: SummaryRepository):
    results = summary_repository.get_by_metric_id(1)

    assert len(results) >= 2
    assert all(s.metric_id == 1 for s in results)


def test_get_with_all_relations(summary_repository: SummaryRepository):
    result = summary_repository.get_with_all_relations(1, 1)

    assert result is not None
    assert hasattr(result, "pipeline")
    assert hasattr(result, "metric")
    assert result.pipeline is not None
    assert result.metric is not None


def test_get_by_metric_result_range(summary_repository: SummaryRepository):
    results = summary_repository.get_by_metric_result_range(1, 0.8, 0.9)

    assert len(results) >= 2
    assert all(0.8 <= s.metric_result <= 0.9 for s in results)


def test_get_top_pipelines_by_metric(summary_repository: SummaryRepository):
    results = summary_repository.get_top_pipelines_by_metric(1, limit=5, ascending=False)

    assert len(results) >= 1
    for i in range(len(results) - 1):
        assert results[i].metric_result >= results[i + 1].metric_result


def test_get_pipeline_summaries_with_relations(summary_repository: SummaryRepository):
    results = summary_repository.get_pipeline_summaries_with_relations(1)

    assert len(results) >= 1
    for summary in results:
        assert hasattr(summary, "pipeline")
        assert hasattr(summary, "metric")


def test_get_metric_summaries_with_relations(summary_repository: SummaryRepository):
    results = summary_repository.get_metric_summaries_with_relations(1)

    assert len(results) >= 2
    for summary in results:
        assert hasattr(summary, "pipeline")
        assert hasattr(summary, "metric")


def test_compare_pipelines_by_metric(summary_repository: SummaryRepository):
    results = summary_repository.compare_pipelines_by_metric([1, 2], 1)

    assert len(results) >= 2
    assert all(s.metric_id == 1 for s in results)
    assert any(s.pipeline_id == 1 for s in results)
    assert any(s.pipeline_id == 2 for s in results)


def test_delete_by_composite_key(summary_repository: SummaryRepository, db_session: Session):
    new_summary = Summary(
        pipeline_id=2,
        metric_id=2,
        metric_result=0.95,
        token_usage=100,
        execution_time=1000,
        result_metadata={"test": "delete"},
    )
    summary_repository.add(new_summary)
    db_session.flush()

    deleted = summary_repository.delete_by_composite_key(2, 2)
    db_session.flush()

    assert deleted is True
    assert summary_repository.get_by_composite_key(2, 2) is None


def test_exists_by_composite_key(summary_repository: SummaryRepository):
    exists = summary_repository.exists_by_composite_key(1, 1)
    assert exists is True

    not_exists = summary_repository.exists_by_composite_key(999, 999)
    assert not_exists is False
