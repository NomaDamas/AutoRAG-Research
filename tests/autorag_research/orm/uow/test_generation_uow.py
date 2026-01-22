import pytest

from autorag_research.orm.repository.chunk import ChunkRepository
from autorag_research.orm.repository.chunk_retrieved_result import ChunkRetrievedResultRepository
from autorag_research.orm.repository.executor_result import ExecutorResultRepository
from autorag_research.orm.repository.pipeline import PipelineRepository
from autorag_research.orm.repository.query import QueryRepository
from autorag_research.orm.uow.generation_uow import GenerationUnitOfWork


@pytest.mark.parametrize(
    ("repo_property", "expected_class"),
    [
        ("queries", QueryRepository),
        ("chunks", ChunkRepository),
        ("chunk_results", ChunkRetrievedResultRepository),
        ("executor_results", ExecutorResultRepository),
        ("pipelines", PipelineRepository),
    ],
)
def test_repository_returns_correct_type(session_factory, repo_property, expected_class):
    with GenerationUnitOfWork(session_factory) as uow:
        repo = getattr(uow, repo_property)
        assert isinstance(repo, expected_class)


def test_can_use_existing_seed_data(session_factory):
    with GenerationUnitOfWork(session_factory) as uow:
        query = uow.queries.get_by_id(1)
        chunk = uow.chunks.get_by_id(1)
        pipeline = uow.pipelines.get_by_id(1)

        assert query is not None
        assert query.contents == "What is Doc One about?"
        assert chunk is not None
        assert chunk.contents == "Chunk 1-1"
        assert pipeline is not None
        assert pipeline.name == "baseline"


def test_get_chunks_by_ids(session_factory):
    with GenerationUnitOfWork(session_factory) as uow:
        chunks = uow.chunks.get_by_ids([1, 2, 3])
        assert len(chunks) == 3
        chunk_ids = {c.id for c in chunks}
        assert chunk_ids == {1, 2, 3}


def test_get_chunks_by_ids_empty_list(session_factory):
    with GenerationUnitOfWork(session_factory) as uow:
        chunks = uow.chunks.get_by_ids([])
        assert chunks == []
