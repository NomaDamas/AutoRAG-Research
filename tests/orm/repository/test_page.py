import pytest
from sqlalchemy.orm import Session

from autorag_research.orm.repository.page import PageRepository


@pytest.fixture
def page_repository(db_session: Session) -> PageRepository:
    return PageRepository(db_session)


def test_get_by_document_id(page_repository: PageRepository):
    results = page_repository.get_by_document_id(1)

    assert len(results) >= 2
    assert all(p.document_id == 1 for p in results)
    assert results[0].page_num < results[1].page_num


def test_get_by_document_and_page_num(page_repository: PageRepository):
    result = page_repository.get_by_document_and_page_num(1, 1)

    assert result is not None
    assert result.document_id == 1
    assert result.page_num == 1


def test_get_with_document(page_repository: PageRepository):
    result = page_repository.get_with_document(1)

    assert result is not None
    assert hasattr(result, "document")
    assert result.document is not None


def test_get_with_captions(page_repository: PageRepository):
    result = page_repository.get_with_captions(1)

    assert result is not None
    assert hasattr(result, "captions")


def test_get_with_image_chunks(page_repository: PageRepository):
    result = page_repository.get_with_image_chunks(1)

    assert result is not None
    assert hasattr(result, "image_chunks")


def test_get_with_image_file(page_repository: PageRepository):
    result = page_repository.get_with_image_file(1)

    assert result is not None
    assert hasattr(result, "image_file")


def test_get_all_with_document(page_repository: PageRepository):
    results = page_repository.get_all_with_document(limit=10, offset=0)

    assert len(results) >= 5
    for page in results:
        assert hasattr(page, "document")


def test_search_by_metadata(page_repository: PageRepository):
    results = page_repository.search_by_metadata("dpi", "300")

    assert len(results) >= 1
    assert all(p.page_metadata.get("dpi") == 300 for p in results if p.page_metadata)


def test_get_by_image_path_id(page_repository: PageRepository):
    result = page_repository.get_by_image_path_id(6)

    assert result is not None
    assert result.image_path == 6


def test_count_by_document(page_repository: PageRepository):
    count = page_repository.count_by_document(1)

    assert count == 2


def test_get_page_range(page_repository: PageRepository):
    results = page_repository.get_page_range(1, 1, 2)

    assert len(results) == 2
    assert all(p.document_id == 1 for p in results)
    assert results[0].page_num == 1
    assert results[1].page_num == 2

