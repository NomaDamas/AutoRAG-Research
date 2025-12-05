import pytest

from autorag_research.exceptions import SessionNotSetError
from autorag_research.orm.repository.chunk import ChunkRepository
from autorag_research.orm.repository.query import QueryRepository
from autorag_research.orm.repository.retrieval_relation import RetrievalRelationRepository
from autorag_research.orm.uow import TextOnlyUnitOfWork


def test_queries_repository_returns_query_repository(session_factory):
    with TextOnlyUnitOfWork(session_factory) as uow:
        repo = uow.queries

        assert isinstance(repo, QueryRepository)


def test_chunks_repository_returns_chunk_repository(session_factory):
    with TextOnlyUnitOfWork(session_factory) as uow:
        repo = uow.chunks

        assert isinstance(repo, ChunkRepository)


def test_retrieval_relations_repository_returns_repository(session_factory):
    with TextOnlyUnitOfWork(session_factory) as uow:
        repo = uow.retrieval_relations

        assert isinstance(repo, RetrievalRelationRepository)


def test_repository_lazy_initialization(session_factory):
    with TextOnlyUnitOfWork(session_factory) as uow:
        assert uow._query_repo is None
        _ = uow.queries
        assert uow._query_repo is not None

        first_repo = uow.queries
        second_repo = uow.queries
        assert first_repo is second_repo


def test_repository_access_without_session_raises_error(session_factory):
    uow = TextOnlyUnitOfWork(session_factory)

    with pytest.raises(SessionNotSetError):
        _ = uow.queries


def test_commit(session_factory):
    with TextOnlyUnitOfWork(session_factory) as uow:
        uow.commit()


def test_rollback(session_factory):
    with TextOnlyUnitOfWork(session_factory) as uow:
        uow.rollback()


def test_flush(session_factory):
    with TextOnlyUnitOfWork(session_factory) as uow:
        uow.flush()


def test_repository_reset_after_exit(session_factory):
    uow = TextOnlyUnitOfWork(session_factory)

    with uow:
        _ = uow.queries
        assert uow._query_repo is not None

    assert uow._query_repo is None
    assert uow._chunk_repo is None
    assert uow._retrieval_relation_repo is None


def test_can_use_existing_seed_data(session_factory):
    with TextOnlyUnitOfWork(session_factory) as uow:
        query = uow.queries.get_by_id(1)
        chunk = uow.chunks.get_by_id(1)
        relations = uow.retrieval_relations.get_by_query_id(1)

        assert query is not None
        assert query.contents == "What is Doc One about?"
        assert chunk is not None
        assert chunk.contents == "Chunk 1-1"
        assert len(relations) >= 1
