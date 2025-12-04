import pytest
from sqlalchemy.orm import Session

from autorag_research.orm.repository.pipeline import PipelineRepository


@pytest.fixture
def pipeline_repository(db_session: Session) -> PipelineRepository:
    return PipelineRepository(db_session)


def test_get_with_executor_results(pipeline_repository: PipelineRepository):
    result = pipeline_repository.get_with_executor_results(1)

    assert result is not None
    assert hasattr(result, "executor_results")


def test_get_with_summaries(pipeline_repository: PipelineRepository):
    result = pipeline_repository.get_with_summaries(1)

    assert result is not None
    assert hasattr(result, "summaries")


def test_get_with_retrieved_results(pipeline_repository: PipelineRepository):
    result = pipeline_repository.get_with_retrieved_results(1)

    assert result is not None
    assert hasattr(result, "chunk_retrieved_results")
    assert hasattr(result, "image_chunk_retrieved_results")


def test_get_with_all_relations(pipeline_repository: PipelineRepository):
    result = pipeline_repository.get_with_all_relations(1)

    assert result is not None
    assert hasattr(result, "executor_results")
    assert hasattr(result, "summaries")
    assert hasattr(result, "chunk_retrieved_results")
    assert hasattr(result, "image_chunk_retrieved_results")


def test_get_by_name(pipeline_repository: PipelineRepository):
    result = pipeline_repository.get_by_name("baseline")

    assert result is not None
    assert result.name == "baseline"


def test_get_by_name_not_found(pipeline_repository: PipelineRepository):
    result = pipeline_repository.get_by_name("nonexistent_pipeline")

    assert result is None


def test_search_by_name(pipeline_repository: PipelineRepository):
    results = pipeline_repository.search_by_name("base")

    assert len(results) >= 1
    assert any(p.name == "baseline" for p in results)


def test_get_all_ordered_by_name(pipeline_repository: PipelineRepository):
    results = pipeline_repository.get_all_ordered_by_name()

    assert len(results) >= 2
    names = [p.name for p in results]
    assert names == sorted(names)


def test_exists_by_name(pipeline_repository: PipelineRepository):
    assert pipeline_repository.exists_by_name("baseline") is True
    assert pipeline_repository.exists_by_name("nonexistent_pipeline") is False


def test_get_by_config_key(pipeline_repository: PipelineRepository):
    results = pipeline_repository.get_by_config_key("k", 5)

    assert len(results) >= 1
    assert all(p.config.get("k") == 5 for p in results)
