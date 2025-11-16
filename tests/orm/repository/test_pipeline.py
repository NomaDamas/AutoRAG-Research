import pytest
from sqlalchemy.orm import Session

from autorag_research.orm.repository.pipeline import PipelineRepository


@pytest.fixture
def pipeline_repository(db_session: Session) -> PipelineRepository:
    return PipelineRepository(db_session)


def test_get_by_name(pipeline_repository: PipelineRepository):
    result = pipeline_repository.get_by_name("baseline")

    assert result is not None
    assert result.name == "baseline"
    assert result.config == {"k": 5}


def test_get_with_experiment_results(pipeline_repository: PipelineRepository):
    result = pipeline_repository.get_with_experiment_results(1)

    assert result is not None
    assert hasattr(result, "experiment_results")


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
    assert hasattr(result, "experiment_results")
    assert hasattr(result, "summaries")
    assert hasattr(result, "chunk_retrieved_results")
    assert hasattr(result, "image_chunk_retrieved_results")


def test_search_by_name(pipeline_repository: PipelineRepository):
    results = pipeline_repository.search_by_name("base", limit=10)

    assert len(results) >= 1
    assert any("base" in p.name.lower() for p in results)


def test_get_all_ordered_by_name(pipeline_repository: PipelineRepository):
    results = pipeline_repository.get_all_ordered_by_name(limit=10, offset=0)

    assert len(results) >= 2
    for i in range(len(results) - 1):
        assert results[i].name <= results[i + 1].name


def test_search_by_config(pipeline_repository: PipelineRepository):
    results = pipeline_repository.search_by_config("k", "5")

    assert len(results) >= 1
    assert all(p.config.get("k") == 5 for p in results if p.config)


def test_exists_by_name(pipeline_repository: PipelineRepository):
    exists = pipeline_repository.exists_by_name("baseline")
    assert exists is True

    not_exists = pipeline_repository.exists_by_name("nonexistent_pipeline")
    assert not_exists is False
