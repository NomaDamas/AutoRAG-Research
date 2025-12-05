import pytest

from autorag_research.orm.repository.chunk import ChunkRepository
from autorag_research.orm.repository.chunk_retrieved_result import (
    ChunkRetrievedResultRepository,
)
from autorag_research.orm.repository.image_chunk import ImageChunkRepository
from autorag_research.orm.repository.metric import MetricRepository
from autorag_research.orm.repository.pipeline import PipelineRepository
from autorag_research.orm.repository.query import QueryRepository
from autorag_research.orm.uow import RetrievalUnitOfWork


@pytest.mark.parametrize(
    ("repo_property", "expected_class"),
    [
        ("queries", QueryRepository),
        ("chunks", ChunkRepository),
        ("image_chunks", ImageChunkRepository),
        ("pipelines", PipelineRepository),
        ("metrics", MetricRepository),
        ("chunk_results", ChunkRetrievedResultRepository),
    ],
)
def test_repository_returns_correct_type(session_factory, repo_property, expected_class):
    with RetrievalUnitOfWork(session_factory) as uow:
        repo = getattr(uow, repo_property)
        assert isinstance(repo, expected_class)


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
