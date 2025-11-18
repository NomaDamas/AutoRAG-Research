import pytest
from sqlalchemy.orm import Session

from autorag_research.orm.repository.evaluator_result import EvaluatorResultRepository
from autorag_research.orm.schema import EvaluationResult


@pytest.fixture
def evaluator_result_repository(db_session: Session) -> EvaluatorResultRepository:
    return EvaluatorResultRepository(db_session)


def test_get_by_composite_key(evaluator_result_repository: EvaluatorResultRepository):
    result = evaluator_result_repository.get_by_composite_key(1, 1, 1)

    assert result is not None
    assert result.query_id == 1
    assert result.pipeline_id == 1
    assert result.metric_id == 1
    assert result.metric_result == 0.8


def test_get_by_query_id(evaluator_result_repository: EvaluatorResultRepository):
    results = evaluator_result_repository.get_by_query_id(1)

    assert len(results) >= 2
    assert all(r.query_id == 1 for r in results)


def test_get_by_pipeline_id(evaluator_result_repository: EvaluatorResultRepository):
    results = evaluator_result_repository.get_by_pipeline_id(1)

    assert len(results) >= 2
    assert all(r.pipeline_id == 1 for r in results)


def test_get_by_metric_id(evaluator_result_repository: EvaluatorResultRepository):
    results = evaluator_result_repository.get_by_metric_id(1)

    assert len(results) >= 2
    assert all(r.metric_id == 1 for r in results)


def test_get_by_query_and_pipeline(evaluator_result_repository: EvaluatorResultRepository):
    results = evaluator_result_repository.get_by_query_and_pipeline(1, 1)

    assert len(results) >= 1
    assert all(r.query_id == 1 and r.pipeline_id == 1 for r in results)


def test_get_with_all_relations(evaluator_result_repository: EvaluatorResultRepository):
    result = evaluator_result_repository.get_with_all_relations(1, 1, 1)

    assert result is not None
    assert hasattr(result, "query_obj")
    assert hasattr(result, "pipeline")
    assert hasattr(result, "metric")
    assert result.query_obj is not None
    assert result.pipeline is not None
    assert result.metric is not None


def test_get_with_non_null_metric_result(evaluator_result_repository: EvaluatorResultRepository):
    result = evaluator_result_repository.get_with_non_null_metric_result(1, 1, 1)

    assert result is not None
    assert result.metric_result is not None
    assert result.metric_result == 0.8


def test_get_with_non_null_metric_result_returns_none_for_null(
    evaluator_result_repository: EvaluatorResultRepository, db_session: Session
):
    # Create a result with null metric_result
    new_result = EvaluationResult(
        query_id=5,
        pipeline_id=2,
        metric_id=1,
        metric_result=None,
    )
    evaluator_result_repository.add(new_result)
    db_session.flush()

    result = evaluator_result_repository.get_with_non_null_metric_result(5, 2, 1)
    assert result is None


def test_get_by_metric_result_range(evaluator_result_repository: EvaluatorResultRepository):
    results = evaluator_result_repository.get_by_metric_result_range(1, 0.7, 0.85)

    assert len(results) >= 1
    assert all(r is not None for r in results)
    assert all(0.7 <= r.metric_result <= 0.85 for r in results if r.metric_result is not None)


def test_get_by_pipeline_and_metric(evaluator_result_repository: EvaluatorResultRepository):
    results = evaluator_result_repository.get_by_pipeline_and_metric(1, 1)

    assert len(results) >= 1
    assert all(r.pipeline_id == 1 and r.metric_id == 1 for r in results)


def test_exists_by_composite_key(evaluator_result_repository: EvaluatorResultRepository):
    exists = evaluator_result_repository.exists_by_composite_key(1, 1, 1)

    assert exists is True

    not_exists = evaluator_result_repository.exists_by_composite_key(999, 999, 999)
    assert not_exists is False


def test_delete_by_composite_key(evaluator_result_repository: EvaluatorResultRepository, db_session: Session):
    new_result = EvaluationResult(
        query_id=4,
        pipeline_id=2,
        metric_id=2,
        metric_result=0.95,
    )
    evaluator_result_repository.add(new_result)
    db_session.flush()

    deleted = evaluator_result_repository.delete_by_composite_key(4, 2, 2)
    db_session.flush()

    assert deleted is True
    assert evaluator_result_repository.get_by_composite_key(4, 2, 2) is None


def test_delete_by_composite_key_returns_false_for_nonexistent(evaluator_result_repository: EvaluatorResultRepository):
    deleted = evaluator_result_repository.delete_by_composite_key(999, 999, 999)

    assert deleted is False
