import pytest
from sqlalchemy.orm import Session

from autorag_research.orm.repository.executor_result import ExecutorResultRepository
from autorag_research.orm.schema import ExperimentResult


@pytest.fixture
def experiment_result_repository(db_session: Session) -> ExecutorResultRepository:
    return ExecutorResultRepository(db_session)


def test_get_by_composite_key(experiment_result_repository: ExecutorResultRepository):
    result = experiment_result_repository.get_by_composite_key(1, 1, 1)

    assert result is not None
    assert result.query_id == 1
    assert result.pipeline_id == 1
    assert result.metric_id == 1
    assert result.metric_result == 0.8


def test_get_by_query_id(experiment_result_repository: ExecutorResultRepository):
    results = experiment_result_repository.get_by_query_id(1)

    assert len(results) >= 2
    assert all(r.query_id == 1 for r in results)


def test_get_by_pipeline_id(experiment_result_repository: ExecutorResultRepository):
    results = experiment_result_repository.get_by_pipeline_id(1)

    assert len(results) >= 2
    assert all(r.pipeline_id == 1 for r in results)


def test_get_by_metric_id(experiment_result_repository: ExecutorResultRepository):
    results = experiment_result_repository.get_by_metric_id(1)

    assert len(results) >= 2
    assert all(r.metric_id == 1 for r in results)


def test_get_by_query_and_pipeline(experiment_result_repository: ExecutorResultRepository):
    results = experiment_result_repository.get_by_query_and_pipeline(1, 1)

    assert len(results) >= 1
    assert all(r.query_id == 1 and r.pipeline_id == 1 for r in results)


def test_get_by_query_and_metric(experiment_result_repository: ExecutorResultRepository):
    results = experiment_result_repository.get_by_query_and_metric(1, 1)

    assert len(results) >= 2
    assert all(r.query_id == 1 and r.metric_id == 1 for r in results)


def test_get_by_pipeline_and_metric(experiment_result_repository: ExecutorResultRepository):
    results = experiment_result_repository.get_by_pipeline_and_metric(1, 1)

    assert len(results) >= 1
    assert all(r.pipeline_id == 1 and r.metric_id == 1 for r in results)


def test_get_with_all_relations(experiment_result_repository: ExecutorResultRepository):
    result = experiment_result_repository.get_with_all_relations(1, 1, 1)

    assert result is not None
    assert hasattr(result, "query_obj")
    assert hasattr(result, "pipeline")
    assert hasattr(result, "metric")
    assert result.query_obj is not None
    assert result.pipeline is not None
    assert result.metric is not None


def test_get_with_generation_results(experiment_result_repository: ExecutorResultRepository):
    results = experiment_result_repository.get_with_generation_results(2, 1)

    assert len(results) >= 1
    assert all(r.generation_result is not None for r in results)
    assert any(r.generation_result == "Generated text 1" for r in results)


def test_get_by_metric_result_range(experiment_result_repository: ExecutorResultRepository):
    results = experiment_result_repository.get_by_metric_result_range(1, 1, 0.7, 0.85)

    assert len(results) >= 1
    assert all(r is not None for r in results)
    assert all(0.7 <= r.metric_result <= 0.85 for r in results)  # ty: ignore


def test_exists_by_composite_key(experiment_result_repository: ExecutorResultRepository):
    exists = experiment_result_repository.exists_by_composite_key(1, 1, 1)

    assert exists is True

    not_exists = experiment_result_repository.exists_by_composite_key(999, 999, 999)
    assert not_exists is False


def test_delete_by_composite_key(experiment_result_repository: ExecutorResultRepository, db_session: Session):
    new_result = ExperimentResult(
        query_id=1,
        pipeline_id=2,
        metric_id=2,
        generation_result="test delete",
        metric_result=0.99,
        token_usage=50,
        execution_time=500,
        result_metadata={"test": "delete"},
    )
    experiment_result_repository.add(new_result)
    db_session.flush()

    deleted = experiment_result_repository.delete_by_composite_key(1, 2, 2)
    db_session.flush()

    assert deleted is True
    assert experiment_result_repository.get_by_composite_key(1, 2, 2) is None
