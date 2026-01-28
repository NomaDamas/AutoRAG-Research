import pytest

from autorag_research.exceptions import DuplicateRetrievalGTError, LengthMismatchError
from autorag_research.orm.models.retrieval_gt import TextId, text
from autorag_research.orm.schema import Chunk, Query, RetrievalRelation
from autorag_research.orm.service.base_ingestion import BaseIngestionService
from autorag_research.orm.uow.text_uow import TextOnlyUnitOfWork


class ConcreteTestIngestionService(BaseIngestionService):
    def _create_uow(self) -> TextOnlyUnitOfWork:
        return TextOnlyUnitOfWork(self.session_factory, self._schema)

    def _get_schema_classes(self) -> dict[str, type]:
        return {
            "Query": Query,
            "Chunk": Chunk,
            "RetrievalRelation": RetrievalRelation,
        }


class TestBaseIngestionService:
    @pytest.fixture
    def service(self, session_factory):
        return ConcreteTestIngestionService(session_factory)

    def test_add_chunks(self, service):
        chunks = [
            {"contents": "test chunk A", "parent_page": None},
            {"contents": "test chunk B", "parent_page": None},
        ]
        ids = service.add_chunks(chunks)
        assert len(ids) == 2
        assert all(isinstance(id_, int) for id_ in ids)

        with service._create_uow() as uow:
            for id_ in ids:
                chunk = uow.chunks.get_by_id(id_)
                assert chunk is not None
                uow.chunks.delete(chunk)
            uow.commit()

    def test_add_queries(self, service):
        queries = [
            {"contents": "test query 1", "generation_gt": ["answer1"]},
            {"contents": "test query 2", "generation_gt": None},
        ]
        ids = service.add_queries(queries)
        assert len(ids) == 2
        assert all(isinstance(id_, int) for id_ in ids)

        with service._create_uow() as uow:
            for id_ in ids:
                query = uow.queries.get_by_id(id_)
                assert query is not None
                uow.queries.delete(query)
            uow.commit()

    def test_set_query_embeddings(self, service):
        query_ids = [1, 2]
        embeddings = [[0.1] * 768, [0.2] * 768]
        updated = service.set_query_embeddings(query_ids, embeddings)
        assert updated == 2

        with service._create_uow() as uow:
            for qid in query_ids:
                query = uow.queries.get_by_id(qid)
                query.embedding = None
            uow.commit()

    def test_set_query_embeddings_length_mismatch(self, service):
        with pytest.raises(LengthMismatchError):
            service.set_query_embeddings([1, 2], [[0.1] * 768])

    def test_set_chunk_embeddings(self, service):
        chunk_ids = [1, 2]
        embeddings = [[0.3] * 768, [0.4] * 768]
        updated = service.set_chunk_embeddings(chunk_ids, embeddings)
        assert updated == 2

        with service._create_uow() as uow:
            for cid in chunk_ids:
                chunk = uow.chunks.get_by_id(cid)
                chunk.embedding = None
            uow.commit()

    def test_set_query_multi_embeddings(self, service):
        query_ids = [1, 2]
        # Use 768 dimensions to match schema
        multi_embeddings = [[[0.1] * 768] * 3, [[0.2] * 768] * 3]
        updated = service.set_query_multi_embeddings(query_ids, multi_embeddings)
        assert updated == 2

        with service._create_uow() as uow:
            for qid in query_ids:
                query = uow.queries.get_by_id(qid)
                query.embeddings = None
            uow.commit()

    def test_set_chunk_multi_embeddings(self, service):
        chunk_ids = [1, 2]
        # Use 768 dimensions to match schema
        multi_embeddings = [[[0.3] * 768] * 3, [[0.4] * 768] * 3]
        updated = service.set_chunk_multi_embeddings(chunk_ids, multi_embeddings)
        assert updated == 2

        with service._create_uow() as uow:
            for cid in chunk_ids:
                chunk = uow.chunks.get_by_id(cid)
                chunk.embeddings = None
            uow.commit()

    def test_add_retrieval_gt_text_mode(self, service):
        # Test upsert=True overwrites existing relations
        pks = service.add_retrieval_gt(query_id=1, gt=text(1) | text(2), chunk_type="text", upsert=True)
        assert len(pks) == 2
        assert all(pk[0] == 1 for pk in pks)

        # Test upsert=False raises error when relation already exists
        with pytest.raises(DuplicateRetrievalGTError):
            service.add_retrieval_gt(query_id=1, gt=text(1), chunk_type="text", upsert=False)

        with service._create_uow() as uow:
            uow.retrieval_relations.delete_by_query_id(1)
            original_rel = RetrievalRelation(query_id=1, group_index=0, group_order=0, chunk_id=1, image_chunk_id=None)
            uow.retrieval_relations.add(original_rel)
            uow.commit()

    def test_add_retrieval_gt_mixed_mode(self, service):
        # Test upsert=True overwrites existing relations
        pks = service.add_retrieval_gt(query_id=2, gt=TextId(3), chunk_type="mixed", upsert=True)
        assert len(pks) == 1
        assert pks[0][0] == 2

        # Test upsert=False raises error when relation already exists
        with pytest.raises(DuplicateRetrievalGTError):
            service.add_retrieval_gt(query_id=2, gt=TextId(3), chunk_type="mixed", upsert=False)

        with service._create_uow() as uow:
            uow.retrieval_relations.delete_by_query_id(2)
            original_rel = RetrievalRelation(query_id=2, group_index=0, group_order=0, chunk_id=3, image_chunk_id=None)
            uow.retrieval_relations.add(original_rel)
            uow.commit()

    def test_add_retrieval_gt_batch(self, service):
        items = [
            (3, text(4)),
            (4, text(5)),
        ]
        # Test upsert=True overwrites existing relations
        pks = service.add_retrieval_gt_batch(items, chunk_type="text", upsert=True)
        assert len(pks) == 2

        # Test upsert=False raises error when relation already exists
        with pytest.raises(DuplicateRetrievalGTError):
            service.add_retrieval_gt_batch(items, chunk_type="text", upsert=False)

        with service._create_uow() as uow:
            uow.retrieval_relations.delete_by_query_id(3)
            uow.retrieval_relations.delete_by_query_id(4)
            original_rel_3 = RetrievalRelation(
                query_id=3, group_index=0, group_order=0, chunk_id=None, image_chunk_id=1
            )
            original_rel_4 = RetrievalRelation(
                query_id=4, group_index=0, group_order=0, chunk_id=None, image_chunk_id=2
            )
            uow.retrieval_relations.add(original_rel_3)
            uow.retrieval_relations.add(original_rel_4)
            uow.commit()
