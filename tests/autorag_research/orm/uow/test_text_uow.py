import pytest

from autorag_research.orm.repository.chunk import ChunkRepository
from autorag_research.orm.repository.query import QueryRepository
from autorag_research.orm.repository.retrieval_relation import RetrievalRelationRepository
from autorag_research.orm.uow import TextOnlyUnitOfWork


@pytest.mark.parametrize(
    ("repo_property", "expected_class"),
    [
        ("queries", QueryRepository),
        ("chunks", ChunkRepository),
        ("retrieval_relations", RetrievalRelationRepository),
    ],
)
def test_repository_returns_correct_type(session_factory, repo_property, expected_class):
    with TextOnlyUnitOfWork(session_factory) as uow:
        repo = getattr(uow, repo_property)
        assert isinstance(repo, expected_class)


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
