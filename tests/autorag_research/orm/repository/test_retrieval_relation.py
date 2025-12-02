import pytest
from sqlalchemy.orm import Session

from autorag_research.orm.repository.retrieval_relation import RetrievalRelationRepository
from autorag_research.orm.schema import Query, RetrievalRelation


@pytest.fixture
def retrieval_relation_repo(db_session: Session) -> RetrievalRelationRepository:
    return RetrievalRelationRepository(db_session)


class TestGetByQueryId:
    def test_get_by_query_id_with_text_chunk(self, retrieval_relation_repo: RetrievalRelationRepository):
        relations = retrieval_relation_repo.get_by_query_id(1)

        assert len(relations) >= 1
        assert all(r.query_id == 1 for r in relations)
        assert relations[0].chunk_id == 1
        assert relations[0].image_chunk_id is None

    def test_get_by_query_id_with_image_chunk(self, retrieval_relation_repo: RetrievalRelationRepository):
        relations = retrieval_relation_repo.get_by_query_id(3)

        assert len(relations) >= 1
        assert all(r.query_id == 3 for r in relations)
        assert relations[0].image_chunk_id == 1
        assert relations[0].chunk_id is None

    def test_get_by_query_id_not_found(self, retrieval_relation_repo: RetrievalRelationRepository):
        relations = retrieval_relation_repo.get_by_query_id(999999)

        assert relations == []

    def test_get_by_query_id_ordered(self, retrieval_relation_repo: RetrievalRelationRepository, db_session: Session):
        query = Query(contents="Test query for ordering")
        db_session.add(query)
        db_session.flush()

        relations_data = [
            RetrievalRelation(query_id=query.id, chunk_id=1, group_index=1, group_order=1),
            RetrievalRelation(query_id=query.id, chunk_id=2, group_index=0, group_order=0),
            RetrievalRelation(query_id=query.id, chunk_id=3, group_index=1, group_order=0),
        ]
        db_session.add_all(relations_data)
        db_session.flush()

        relations = retrieval_relation_repo.get_by_query_id(query.id)

        assert len(relations) == 3
        assert relations[0].group_index == 0
        assert relations[0].group_order == 0
        assert relations[1].group_index == 1
        assert relations[1].group_order == 0
        assert relations[2].group_index == 1
        assert relations[2].group_order == 1


class TestGetByQueryAndChunk:
    def test_get_by_query_and_chunk(self, retrieval_relation_repo: RetrievalRelationRepository):
        result = retrieval_relation_repo.get_by_query_and_chunk(1, 1)

        assert result is not None
        assert result.query_id == 1
        assert result.chunk_id == 1
        assert result.image_chunk_id is None

    def test_get_by_query_and_chunk_not_found(self, retrieval_relation_repo: RetrievalRelationRepository):
        result = retrieval_relation_repo.get_by_query_and_chunk(1, 999999)

        assert result is None


class TestGetByQueryAndImageChunk:
    def test_get_by_query_and_image_chunk(self, retrieval_relation_repo: RetrievalRelationRepository):
        result = retrieval_relation_repo.get_by_query_and_image_chunk(3, 1)

        assert result is not None
        assert result.query_id == 3
        assert result.image_chunk_id == 1
        assert result.chunk_id is None

    def test_get_by_query_and_image_chunk_not_found(self, retrieval_relation_repo: RetrievalRelationRepository):
        result = retrieval_relation_repo.get_by_query_and_image_chunk(3, 999999)

        assert result is None


class TestGetMaxGroupIndex:
    def test_get_max_group_index(self, retrieval_relation_repo: RetrievalRelationRepository):
        result = retrieval_relation_repo.get_max_group_index(1)

        assert result == 0

    def test_get_max_group_index_multiple_groups(
        self, retrieval_relation_repo: RetrievalRelationRepository, db_session: Session
    ):
        query = Query(contents="Test query for max group index")
        db_session.add(query)
        db_session.flush()

        relations_data = [
            RetrievalRelation(query_id=query.id, chunk_id=1, group_index=0, group_order=0),
            RetrievalRelation(query_id=query.id, chunk_id=2, group_index=2, group_order=0),
            RetrievalRelation(query_id=query.id, chunk_id=3, group_index=5, group_order=0),
        ]
        db_session.add_all(relations_data)
        db_session.flush()

        result = retrieval_relation_repo.get_max_group_index(query.id)

        assert result == 5

    def test_get_max_group_index_not_found(self, retrieval_relation_repo: RetrievalRelationRepository):
        result = retrieval_relation_repo.get_max_group_index(999999)

        assert result is None


class TestGetMaxGroupOrder:
    def test_get_max_group_order(self, retrieval_relation_repo: RetrievalRelationRepository):
        result = retrieval_relation_repo.get_max_group_order(1, 0)

        assert result == 0

    def test_get_max_group_order_multiple_orders(
        self, retrieval_relation_repo: RetrievalRelationRepository, db_session: Session
    ):
        query = Query(contents="Test query for max group order")
        db_session.add(query)
        db_session.flush()

        relations_data = [
            RetrievalRelation(query_id=query.id, chunk_id=1, group_index=0, group_order=0),
            RetrievalRelation(query_id=query.id, chunk_id=2, group_index=0, group_order=1),
            RetrievalRelation(query_id=query.id, chunk_id=3, group_index=0, group_order=5),
        ]
        db_session.add_all(relations_data)
        db_session.flush()

        result = retrieval_relation_repo.get_max_group_order(query.id, 0)

        assert result == 5

    def test_get_max_group_order_not_found(self, retrieval_relation_repo: RetrievalRelationRepository):
        result = retrieval_relation_repo.get_max_group_order(999999, 0)

        assert result is None


class TestCountByQuery:
    def test_count_by_query(self, retrieval_relation_repo: RetrievalRelationRepository):
        result = retrieval_relation_repo.count_by_query(1)

        assert result >= 1

    def test_count_by_query_not_found(self, retrieval_relation_repo: RetrievalRelationRepository):
        result = retrieval_relation_repo.count_by_query(999999)

        assert result == 0


class TestCountTextChunksByQuery:
    def test_count_text_chunks_by_query(self, retrieval_relation_repo: RetrievalRelationRepository):
        result = retrieval_relation_repo.count_text_chunks_by_query(1)

        assert result >= 1

    def test_count_text_chunks_by_query_image_only(self, retrieval_relation_repo: RetrievalRelationRepository):
        result = retrieval_relation_repo.count_text_chunks_by_query(3)

        assert result == 0


class TestCountImageChunksByQuery:
    def test_count_image_chunks_by_query(self, retrieval_relation_repo: RetrievalRelationRepository):
        result = retrieval_relation_repo.count_image_chunks_by_query(3)

        assert result >= 1

    def test_count_image_chunks_by_query_text_only(self, retrieval_relation_repo: RetrievalRelationRepository):
        result = retrieval_relation_repo.count_image_chunks_by_query(1)

        assert result == 0


class TestCustomSchema:
    def test_with_custom_schema(self, db_session: Session):
        from autorag_research.orm.schema_factory import create_schema

        custom_schema = create_schema(embedding_dim=1024)
        repo = RetrievalRelationRepository(db_session, custom_schema.RetrievalRelation)

        result = repo.get_by_query_id(1)

        assert len(result) >= 1
