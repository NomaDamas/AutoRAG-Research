import pytest
from sqlalchemy.orm import Session

from autorag_research.orm.repository.image_chunk import ImageChunkRepository


@pytest.fixture
def image_chunk_repository(db_session: Session) -> ImageChunkRepository:
    return ImageChunkRepository(db_session)


def test_get_by_page_id(image_chunk_repository: ImageChunkRepository):
    results = image_chunk_repository.get_by_page_id(1)

    assert len(results) >= 1
    for ic in results:
        assert ic.parent_page == 1


def test_get_with_page(image_chunk_repository: ImageChunkRepository):
    result = image_chunk_repository.get_with_page(1)

    assert result is not None
    assert hasattr(result, "page")
    assert result.page is not None


def test_get_with_retrieval_relations(image_chunk_repository: ImageChunkRepository):
    result = image_chunk_repository.get_with_retrieval_relations(1)

    assert result is not None
    assert hasattr(result, "retrieval_relations")


def test_get_with_image_chunk_retrieved_results(image_chunk_repository: ImageChunkRepository):
    result = image_chunk_repository.get_with_image_chunk_retrieved_results(1)

    assert result is not None
    assert hasattr(result, "image_chunk_retrieved_results")


def test_get_image_chunks_with_embeddings(image_chunk_repository: ImageChunkRepository):
    results = image_chunk_repository.get_with_embeddings(limit=10, offset=0)

    assert isinstance(results, list)
    assert all(ic.embedding is not None for ic in results)


def test_get_image_chunks_without_embeddings(image_chunk_repository: ImageChunkRepository):
    results = image_chunk_repository.get_without_embeddings(limit=10, offset=0)

    assert isinstance(results, list)
    for ic in results:
        assert ic.embedding is None


def test_count_by_page(image_chunk_repository: ImageChunkRepository):
    count = image_chunk_repository.count_by_page(1)

    assert count >= 1


def test_get_with_all_relations(image_chunk_repository: ImageChunkRepository):
    result = image_chunk_repository.get_with_all_relations(1)

    assert result is not None
    assert hasattr(result, "page")
    assert hasattr(result, "retrieval_relations")
    assert hasattr(result, "image_chunk_retrieved_results")
