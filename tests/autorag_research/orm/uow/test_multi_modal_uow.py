import pytest

from autorag_research.orm.repository.caption import CaptionRepository
from autorag_research.orm.repository.chunk import ChunkRepository
from autorag_research.orm.repository.document import DocumentRepository
from autorag_research.orm.repository.file import FileRepository
from autorag_research.orm.repository.image_chunk import ImageChunkRepository
from autorag_research.orm.repository.page import PageRepository
from autorag_research.orm.repository.query import QueryRepository
from autorag_research.orm.repository.retrieval_relation import RetrievalRelationRepository
from autorag_research.orm.uow import MultiModalUnitOfWork


@pytest.mark.parametrize(
    ("repo_property", "expected_class"),
    [
        ("files", FileRepository),
        ("documents", DocumentRepository),
        ("pages", PageRepository),
        ("captions", CaptionRepository),
        ("chunks", ChunkRepository),
        ("image_chunks", ImageChunkRepository),
        ("queries", QueryRepository),
        ("retrieval_relations", RetrievalRelationRepository),
    ],
)
def test_repository_returns_correct_type(session_factory, repo_property, expected_class):
    with MultiModalUnitOfWork(session_factory) as uow:
        repo = getattr(uow, repo_property)
        assert isinstance(repo, expected_class)


def test_can_use_existing_seed_data(session_factory):
    with MultiModalUnitOfWork(session_factory) as uow:
        file = uow.files.get_by_id(1)
        document = uow.documents.get_by_id(1)
        page = uow.pages.get_by_id(1)
        caption = uow.captions.get_by_id(1)
        chunk = uow.chunks.get_by_id(1)
        image_chunk = uow.image_chunks.get_by_id(1)
        query = uow.queries.get_by_id(1)

        assert file is not None
        assert file.path == "/data/doc1.pdf"
        assert document is not None
        assert document.filename == "doc1.pdf"
        assert page is not None
        assert page.page_num == 1
        assert caption is not None
        assert caption.contents == "Caption for page 1 of doc1"
        assert chunk is not None
        assert chunk.contents == "Chunk 1-1"
        assert image_chunk is not None
        assert query is not None
        assert query.contents == "What is Doc One about?"
