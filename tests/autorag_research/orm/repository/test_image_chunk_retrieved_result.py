import pytest
from sqlalchemy.orm import Session

from autorag_research.orm.repository.image_chunk_retrieved_result import ImageChunkRetrievedResultRepository
from autorag_research.orm.schema import ImageChunkRetrievedResult


@pytest.fixture
def image_chunk_retrieved_result_repository(db_session: Session) -> ImageChunkRetrievedResultRepository:
    return ImageChunkRetrievedResultRepository(db_session)


def test_get_by_query_and_pipeline(
    image_chunk_retrieved_result_repository: ImageChunkRetrievedResultRepository,
    db_session: Session,
):
    # Add test data
    test_result = ImageChunkRetrievedResult(query_id=1, pipeline_id=1, image_chunk_id=1, rel_score=0.85)
    db_session.add(test_result)
    db_session.flush()

    results = image_chunk_retrieved_result_repository.get_by_query_and_pipeline([1], 1)

    assert len(results) >= 1
    assert all(r.query_id == 1 and r.pipeline_id == 1 for r in results)

    # Cleanup
    db_session.delete(test_result)
    db_session.commit()


def test_get_by_query_and_pipeline_returns_ordered_by_score(
    image_chunk_retrieved_result_repository: ImageChunkRetrievedResultRepository,
    db_session: Session,
):
    # Add multiple results with different scores
    results_to_add = [
        ImageChunkRetrievedResult(query_id=1, pipeline_id=1, image_chunk_id=1, rel_score=0.5),
        ImageChunkRetrievedResult(query_id=1, pipeline_id=1, image_chunk_id=2, rel_score=0.9),
        ImageChunkRetrievedResult(query_id=1, pipeline_id=1, image_chunk_id=3, rel_score=0.7),
    ]
    db_session.add_all(results_to_add)
    db_session.flush()

    results = image_chunk_retrieved_result_repository.get_by_query_and_pipeline([1], 1)

    assert len(results) == 3
    assert results[0].rel_score == 0.9
    assert results[1].rel_score == 0.7
    assert results[2].rel_score == 0.5

    # Cleanup
    for r in results_to_add:
        db_session.delete(r)
    db_session.commit()


def test_bulk_insert(
    image_chunk_retrieved_result_repository: ImageChunkRetrievedResultRepository,
    db_session: Session,
):
    results_to_insert = [
        {"query_id": 4, "pipeline_id": 2, "image_chunk_id": 1, "rel_score": 0.75},
        {"query_id": 2, "pipeline_id": 2, "image_chunk_id": 2, "rel_score": 0.65},
    ]

    inserted_count = image_chunk_retrieved_result_repository.bulk_insert(results_to_insert)
    db_session.flush()

    assert inserted_count == 2

    results = image_chunk_retrieved_result_repository.get_by_query_and_pipeline([2, 4], 2)
    assert len(results) == 3

    # Cleanup
    image_chunk_retrieved_result_repository.delete_by_query_and_pipeline(4, 2)
    image_chunk_retrieved_result_repository.delete_by_query_and_pipeline(2, 2)
    db_session.commit()
