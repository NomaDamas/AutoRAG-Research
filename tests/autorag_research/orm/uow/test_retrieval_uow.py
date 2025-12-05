import pytest
from sqlalchemy.orm import Session

from autorag_research.exceptions import SessionNotSetError
from autorag_research.orm.repository.chunk import ChunkRepository
from autorag_research.orm.repository.chunk_retrieved_result import ChunkRetrievedResultRepository
from autorag_research.orm.repository.image_chunk import ImageChunkRepository
from autorag_research.orm.repository.metric import MetricRepository
from autorag_research.orm.repository.pipeline import PipelineRepository
from autorag_research.orm.repository.query import QueryRepository
from autorag_research.orm.uow import RetrievalUnitOfWork


def test_context_manager_creates_session(session_factory):
    uow = RetrievalUnitOfWork(session_factory)
    assert uow.session is None

    with uow:
        assert uow.session is not None
        assert isinstance(uow.session, Session)


def test_queries_repository_returns_query_repository(session_factory):
    with RetrievalUnitOfWork(session_factory) as uow:
        repo = uow.queries

        assert isinstance(repo, QueryRepository)


def test_chunks_repository_returns_chunk_repository(session_factory):
    with RetrievalUnitOfWork(session_factory) as uow:
        repo = uow.chunks

        assert isinstance(repo, ChunkRepository)


def test_image_chunks_repository_returns_image_chunk_repository(session_factory):
    with RetrievalUnitOfWork(session_factory) as uow:
        repo = uow.image_chunks

        assert isinstance(repo, ImageChunkRepository)


def test_pipelines_repository_returns_pipeline_repository(session_factory):
    with RetrievalUnitOfWork(session_factory) as uow:
        repo = uow.pipelines

        assert isinstance(repo, PipelineRepository)


def test_metrics_repository_returns_metric_repository(session_factory):
    with RetrievalUnitOfWork(session_factory) as uow:
        repo = uow.metrics

        assert isinstance(repo, MetricRepository)


def test_chunk_results_repository_returns_repository(session_factory):
    with RetrievalUnitOfWork(session_factory) as uow:
        repo = uow.chunk_results

        assert isinstance(repo, ChunkRetrievedResultRepository)


def test_repository_lazy_initialization(session_factory):
    with RetrievalUnitOfWork(session_factory) as uow:
        assert uow._query_repo is None
        _ = uow.queries
        assert uow._query_repo is not None

        first_repo = uow.queries
        second_repo = uow.queries
        assert first_repo is second_repo


def test_repository_access_without_session_raises_error(session_factory):
    uow = RetrievalUnitOfWork(session_factory)

    with pytest.raises(SessionNotSetError):
        _ = uow.queries


def test_commit(session_factory):
    with RetrievalUnitOfWork(session_factory) as uow:
        uow.commit()


def test_rollback(session_factory):
    with RetrievalUnitOfWork(session_factory) as uow:
        uow.rollback()


def test_flush(session_factory):
    with RetrievalUnitOfWork(session_factory) as uow:
        uow.flush()


def test_repository_reset_after_exit(session_factory):
    uow = RetrievalUnitOfWork(session_factory)

    with uow:
        _ = uow.queries
        _ = uow.pipelines
        assert uow._query_repo is not None
        assert uow._pipeline_repo is not None

    assert uow._query_repo is None
    assert uow._chunk_repo is None
    assert uow._image_chunk_repo is None
    assert uow._pipeline_repo is None
    assert uow._metric_repo is None
    assert uow._chunk_result_repo is None


def test_can_use_existing_seed_data(session_factory):
    with RetrievalUnitOfWork(session_factory) as uow:
        query = uow.queries.get_by_id(1)
        chunk = uow.chunks.get_by_id(1)
        image_chunk = uow.image_chunks.get_by_id(1)
        pipeline = uow.pipelines.get_by_id(1)
        metric = uow.metrics.get_by_id(1)
        chunk_results = uow.chunk_results.get_by_query(1)

        assert query is not None
        assert query.contents == "What is Doc One about?"
        assert chunk is not None
        assert chunk.contents == "Chunk 1-1"
        assert image_chunk is not None
        assert pipeline is not None
        assert pipeline.name == "baseline"
        assert metric is not None
        assert metric.name == "retrieval@k"
        assert len(chunk_results) >= 1
