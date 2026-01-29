import pytest

from autorag_research.orm.service.text_ingestion import TextDataIngestionService


class TestTextDataIngestionService:
    @pytest.fixture
    def service(self, session_factory):
        return TextDataIngestionService(session_factory)

    def test_get_retrieval_gt_by_query(self, service):
        relations = service.get_retrieval_gt_by_query(1)
        assert len(relations) >= 1
        assert all(r.query_id == 1 for r in relations)

    def test_get_statistics(self, service):
        stats = service.get_statistics()
        assert "queries" in stats
        assert "chunks" in stats
        assert stats["queries"]["total"] >= 5
        assert stats["chunks"]["total"] >= 6

    def test_clean_removes_empty_content(self, service):
        chunk_ids = service.add_chunks([
            {"contents": "   "},
            {"contents": ""},
        ])
        query_ids = service.add_queries([
            {"contents": "   ", "generation_gt": None},
            {"contents": "", "generation_gt": None},
        ])
        result = service.clean()
        assert result["deleted_chunks"] >= 2
        assert result["deleted_queries"] >= 2

        with service._create_uow() as uow:
            for cid in chunk_ids:
                assert uow.chunks.get_by_id(cid) is None
            for qid in query_ids:
                assert uow.queries.get_by_id(qid) is None

    def test_link_page_to_chunks(self, service, db_session):
        # Use existing page_id=2 and create new chunks to link
        chunk_ids = service.add_chunks([
            {"contents": "Test chunk for link_page_to_chunks 1"},
            {"contents": "Test chunk for link_page_to_chunks 2"},
        ])

        # Link page to chunks (1:N relationship)
        pks = service.link_page_to_chunks(page_id=2, chunk_ids=chunk_ids)

        assert len(pks) == 2
        assert all(pk[0] == 2 for pk in pks)
        assert {pk[1] for pk in pks} == set(chunk_ids)

        # Cleanup
        from autorag_research.orm.schema import PageChunkRelation

        db_session.query(PageChunkRelation).filter(
            PageChunkRelation.page_id == 2, PageChunkRelation.chunk_id.in_(chunk_ids)
        ).delete()
        with service._create_uow() as uow:
            for cid in chunk_ids:
                chunk = uow.chunks.get_by_id(cid)
                if chunk:
                    uow.chunks.delete(chunk)
            uow.commit()

    def test_link_pages_to_chunks(self, service, db_session):
        # Use existing page_ids and create new chunks to link
        chunk_ids = service.add_chunks([
            {"contents": "Test chunk for link_pages_to_chunks 1"},
            {"contents": "Test chunk for link_pages_to_chunks 2"},
            {"contents": "Test chunk for link_pages_to_chunks 3"},
        ])

        # M:N relationship: page 4 -> chunk[0], chunk[1]; page 6 -> chunk[1], chunk[2]
        relations = [
            {"page_id": 4, "chunk_id": chunk_ids[0]},
            {"page_id": 4, "chunk_id": chunk_ids[1]},
            {"page_id": 6, "chunk_id": chunk_ids[1]},
            {"page_id": 6, "chunk_id": chunk_ids[2]},
        ]
        pks = service.link_pages_to_chunks(relations)

        assert len(pks) == 4
        # Verify page 4 has 2 chunks
        page_4_chunks = [pk[1] for pk in pks if pk[0] == 4]
        assert len(page_4_chunks) == 2
        # Verify page 6 has 2 chunks
        page_6_chunks = [pk[1] for pk in pks if pk[0] == 6]
        assert len(page_6_chunks) == 2
        # Verify chunk[1] is linked to both pages (M:N)
        chunk_1_pages = [pk[0] for pk in pks if pk[1] == chunk_ids[1]]
        assert set(chunk_1_pages) == {4, 6}

        # Cleanup
        from autorag_research.orm.schema import PageChunkRelation

        db_session.query(PageChunkRelation).filter(PageChunkRelation.chunk_id.in_(chunk_ids)).delete()
        with service._create_uow() as uow:
            for cid in chunk_ids:
                chunk = uow.chunks.get_by_id(cid)
                if chunk:
                    uow.chunks.delete(chunk)
            uow.commit()
