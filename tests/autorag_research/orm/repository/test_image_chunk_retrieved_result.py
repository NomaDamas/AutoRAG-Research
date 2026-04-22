import pytest
from sqlalchemy.orm import Session

from autorag_research.orm.repository.image_chunk_retrieved_result import ImageChunkRetrievedResultRepository
from autorag_research.orm.schema import ImageChunk, ImageChunkRetrievedResult, Pipeline, Query


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
    test_queries = [Query(contents="bulk insert query 1"), Query(contents="bulk insert query 2")]
    test_pipeline = Pipeline(name="bulk_insert_pipeline", config={"type": "test"})
    test_image_chunks = [
        ImageChunk(contents=b"chunk-1", mimetype="image/png"),
        ImageChunk(contents=b"chunk-2", mimetype="image/png"),
    ]
    db_session.add_all([*test_queries, test_pipeline, *test_image_chunks])
    db_session.flush()

    query_ids = [query.id for query in test_queries]
    pipeline_id = test_pipeline.id
    inserted_keys = {
        (query_ids[0], pipeline_id, test_image_chunks[0].id),
        (query_ids[1], pipeline_id, test_image_chunks[1].id),
    }
    results_to_insert = [
        {
            "query_id": query_ids[0],
            "pipeline_id": pipeline_id,
            "image_chunk_id": test_image_chunks[0].id,
            "rel_score": 0.75,
        },
        {
            "query_id": query_ids[1],
            "pipeline_id": pipeline_id,
            "image_chunk_id": test_image_chunks[1].id,
            "rel_score": 0.65,
        },
    ]

    inserted_count = image_chunk_retrieved_result_repository.bulk_insert(results_to_insert)
    db_session.flush()

    assert inserted_count == 2

    results = image_chunk_retrieved_result_repository.get_by_query_and_pipeline(query_ids, pipeline_id)
    result_keys = {(result.query_id, result.pipeline_id, result.image_chunk_id) for result in results}

    assert len(results) == len(inserted_keys)
    assert result_keys == inserted_keys

    # Cleanup
    for result in results:
        if (result.query_id, result.pipeline_id, result.image_chunk_id) in inserted_keys:
            db_session.delete(result)
    for image_chunk in test_image_chunks:
        db_session.delete(image_chunk)
    db_session.delete(test_pipeline)
    for query in test_queries:
        db_session.delete(query)
    db_session.commit()
