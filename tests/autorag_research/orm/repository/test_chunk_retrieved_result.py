import pytest
from sqlalchemy.orm import Session

from autorag_research.orm.repository.chunk_retrieved_result import ChunkRetrievedResultRepository
from autorag_research.orm.schema import ChunkRetrievedResult


@pytest.fixture
def chunk_retrieved_result_repository(db_session: Session) -> ChunkRetrievedResultRepository:
    return ChunkRetrievedResultRepository(db_session)


def test_get_by_query_and_pipeline(chunk_retrieved_result_repository: ChunkRetrievedResultRepository):
    # Seed data: (query_id=1, pipeline_id=1, chunk_id=1, rel_score=0.85)
    # Plus, (2, 2, 3, 0.4)
    results = chunk_retrieved_result_repository.get_by_query_and_pipeline([1, 2], 1)

    assert len(results) == 1
    assert all(r.query_id == 1 and r.pipeline_id == 1 for r in results)
    assert results[0].rel_score == 0.85


def test_get_by_query_and_pipeline_returns_ordered_by_score(
    chunk_retrieved_result_repository: ChunkRetrievedResultRepository,
    db_session: Session,
):
    # Add multiple results with different scores
    results_to_add = [
        ChunkRetrievedResult(query_id=3, pipeline_id=1, chunk_id=1, rel_score=0.5),
        ChunkRetrievedResult(query_id=3, pipeline_id=1, chunk_id=2, rel_score=0.9),
        ChunkRetrievedResult(query_id=3, pipeline_id=1, chunk_id=3, rel_score=0.7),
    ]
    db_session.add_all(results_to_add)
    db_session.flush()

    results = chunk_retrieved_result_repository.get_by_query_and_pipeline([3], 1)

    assert len(results) == 3
    assert results[0].rel_score == 0.9
    assert results[1].rel_score == 0.7
    assert results[2].rel_score == 0.5

    # Cleanup
    for r in results_to_add:
        db_session.delete(r)
    db_session.commit()


def test_get_by_pipeline(chunk_retrieved_result_repository: ChunkRetrievedResultRepository):
    # Seed data: pipeline_id=1 has (query_id=1, chunk_id=1)
    results = chunk_retrieved_result_repository.get_by_pipeline(1)

    assert len(results) >= 1
    assert all(r.pipeline_id == 1 for r in results)


def test_get_by_pipeline_with_limit(chunk_retrieved_result_repository: ChunkRetrievedResultRepository):
    results = chunk_retrieved_result_repository.get_by_pipeline(1, limit=1)

    assert len(results) <= 1


def test_get_by_query(chunk_retrieved_result_repository: ChunkRetrievedResultRepository):
    # Seed data: query_id=1 has (pipeline_id=1, chunk_id=1)
    results = chunk_retrieved_result_repository.get_by_query(1)

    assert len(results) >= 1
    assert all(r.query_id == 1 for r in results)


def test_delete_by_pipeline(chunk_retrieved_result_repository: ChunkRetrievedResultRepository, db_session: Session):
    # Add test data to delete
    test_results = [
        ChunkRetrievedResult(query_id=4, pipeline_id=1, chunk_id=1, rel_score=0.6),
        ChunkRetrievedResult(query_id=4, pipeline_id=1, chunk_id=2, rel_score=0.7),
    ]
    db_session.add_all(test_results)
    db_session.flush()

    # Verify data exists
    before_count = len(chunk_retrieved_result_repository.get_by_query(4))
    assert before_count == 2

    deleted_count = chunk_retrieved_result_repository.delete_by_query_and_pipeline(4, 1)
    db_session.flush()

    assert deleted_count == 2
    assert len(chunk_retrieved_result_repository.get_by_query(4)) == 0


def test_delete_by_query_and_pipeline(
    chunk_retrieved_result_repository: ChunkRetrievedResultRepository,
    db_session: Session,
):
    # Add test data to delete
    test_result = ChunkRetrievedResult(query_id=5, pipeline_id=2, chunk_id=1, rel_score=0.8)
    db_session.add(test_result)
    db_session.flush()

    deleted_count = chunk_retrieved_result_repository.delete_by_query_and_pipeline(5, 2)
    db_session.flush()

    assert deleted_count == 1
    assert len(chunk_retrieved_result_repository.get_by_query_and_pipeline([5], 2)) == 0


def test_delete_by_query_and_pipeline_returns_zero_for_nonexistent(
    chunk_retrieved_result_repository: ChunkRetrievedResultRepository,
):
    deleted_count = chunk_retrieved_result_repository.delete_by_query_and_pipeline(999, 999)

    assert deleted_count == 0


def test_bulk_insert(chunk_retrieved_result_repository: ChunkRetrievedResultRepository, db_session: Session):
    results_to_insert = [
        {"query_id": 4, "pipeline_id": 2, "chunk_id": 1, "rel_score": 0.75},
        {"query_id": 4, "pipeline_id": 2, "chunk_id": 2, "rel_score": 0.65},
        {"query_id": 4, "pipeline_id": 2, "chunk_id": 3, "rel_score": 0.55},
    ]

    inserted_count = chunk_retrieved_result_repository.bulk_insert(results_to_insert)
    db_session.flush()

    assert inserted_count == 3

    results = chunk_retrieved_result_repository.get_by_query_and_pipeline([4], 2)
    assert len(results) == 3

    # Cleanup
    chunk_retrieved_result_repository.delete_by_query_and_pipeline(4, 2)
    db_session.commit()
