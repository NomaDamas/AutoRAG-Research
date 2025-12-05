import pytest
from sqlalchemy.orm import Session

from autorag_research.exceptions import SessionNotSetError
from autorag_research.orm.repository.caption import CaptionRepository
from autorag_research.orm.repository.chunk import ChunkRepository
from autorag_research.orm.repository.document import DocumentRepository
from autorag_research.orm.repository.file import FileRepository
from autorag_research.orm.repository.image_chunk import ImageChunkRepository
from autorag_research.orm.repository.page import PageRepository
from autorag_research.orm.repository.query import QueryRepository
from autorag_research.orm.repository.retrieval_relation import RetrievalRelationRepository
from autorag_research.orm.uow import MultiModalUnitOfWork


def test_context_manager_creates_session(session_factory):
    uow = MultiModalUnitOfWork(session_factory)
    assert uow.session is None

    with uow:
        assert uow.session is not None
        assert isinstance(uow.session, Session)


def test_files_repository_returns_file_repository(session_factory):
    with MultiModalUnitOfWork(session_factory) as uow:
        repo = uow.files

        assert isinstance(repo, FileRepository)


def test_documents_repository_returns_document_repository(session_factory):
    with MultiModalUnitOfWork(session_factory) as uow:
        repo = uow.documents

        assert isinstance(repo, DocumentRepository)


def test_pages_repository_returns_page_repository(session_factory):
    with MultiModalUnitOfWork(session_factory) as uow:
        repo = uow.pages

        assert isinstance(repo, PageRepository)


def test_captions_repository_returns_caption_repository(session_factory):
    with MultiModalUnitOfWork(session_factory) as uow:
        repo = uow.captions

        assert isinstance(repo, CaptionRepository)


def test_chunks_repository_returns_chunk_repository(session_factory):
    with MultiModalUnitOfWork(session_factory) as uow:
        repo = uow.chunks

        assert isinstance(repo, ChunkRepository)


def test_image_chunks_repository_returns_image_chunk_repository(session_factory):
    with MultiModalUnitOfWork(session_factory) as uow:
        repo = uow.image_chunks

        assert isinstance(repo, ImageChunkRepository)


def test_queries_repository_returns_query_repository(session_factory):
    with MultiModalUnitOfWork(session_factory) as uow:
        repo = uow.queries

        assert isinstance(repo, QueryRepository)


def test_retrieval_relations_repository_returns_repository(session_factory):
    with MultiModalUnitOfWork(session_factory) as uow:
        repo = uow.retrieval_relations

        assert isinstance(repo, RetrievalRelationRepository)


def test_repository_lazy_initialization(session_factory):
    with MultiModalUnitOfWork(session_factory) as uow:
        assert uow._file_repo is None
        _ = uow.files
        assert uow._file_repo is not None

        first_repo = uow.files
        second_repo = uow.files
        assert first_repo is second_repo


def test_repository_access_without_session_raises_error(session_factory):
    uow = MultiModalUnitOfWork(session_factory)

    with pytest.raises(SessionNotSetError):
        _ = uow.files


def test_commit(session_factory):
    with MultiModalUnitOfWork(session_factory) as uow:
        uow.commit()


def test_rollback(session_factory):
    with MultiModalUnitOfWork(session_factory) as uow:
        uow.rollback()


def test_flush(session_factory):
    with MultiModalUnitOfWork(session_factory) as uow:
        uow.flush()


def test_repository_reset_after_exit(session_factory):
    uow = MultiModalUnitOfWork(session_factory)

    with uow:
        _ = uow.files
        _ = uow.documents
        assert uow._file_repo is not None
        assert uow._document_repo is not None

    assert uow._file_repo is None
    assert uow._document_repo is None
    assert uow._page_repo is None
    assert uow._caption_repo is None
    assert uow._chunk_repo is None
    assert uow._image_chunk_repo is None
    assert uow._query_repo is None
    assert uow._retrieval_relation_repo is None


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
