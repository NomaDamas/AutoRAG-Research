import pytest
from sqlalchemy.orm import Session

from autorag_research.orm.repository.executor_result import ExecutorResultRepository
from autorag_research.orm.schema import ExecutorResult


@pytest.fixture
def executor_result_repository(db_session: Session) -> ExecutorResultRepository:
    return ExecutorResultRepository(db_session)


def test_get_by_composite_key(executor_result_repository: ExecutorResultRepository):
    result = executor_result_repository.get_by_composite_key(1, 1)

    assert result is not None
    assert result.query_id == 1
    assert result.pipeline_id == 1


def test_get_by_query_id(executor_result_repository: ExecutorResultRepository):
    results = executor_result_repository.get_by_query_id(1)

    assert len(results) >= 2
    assert all(r.query_id == 1 for r in results)


def test_get_by_pipeline_id(executor_result_repository: ExecutorResultRepository):
    results = executor_result_repository.get_by_pipeline_id(1)

    assert len(results) >= 2
    assert all(r.pipeline_id == 1 for r in results)


def test_get_with_all_relations(executor_result_repository: ExecutorResultRepository):
    result = executor_result_repository.get_with_all_relations(1, 1)

    assert result is not None
    assert hasattr(result, "query_obj")
    assert hasattr(result, "pipeline")
    assert result.query_obj is not None
    assert result.pipeline is not None


def test_exists_by_composite_key(executor_result_repository: ExecutorResultRepository):
    exists = executor_result_repository.exists_by_composite_key(1, 1)

    assert exists is True

    not_exists = executor_result_repository.exists_by_composite_key(999, 999)
    assert not_exists is False


def test_delete_by_composite_key(executor_result_repository: ExecutorResultRepository, db_session: Session):
    new_result = ExecutorResult(
        query_id=2,
        pipeline_id=2,
        generation_result="test delete",
        token_usage=50,
        execution_time=500,
        result_metadata={"test": "delete"},
    )
    executor_result_repository.add(new_result)
    db_session.flush()

    deleted = executor_result_repository.delete_by_composite_key(2, 2)
    db_session.flush()

    assert deleted is True
    assert executor_result_repository.get_by_composite_key(2, 2) is None
