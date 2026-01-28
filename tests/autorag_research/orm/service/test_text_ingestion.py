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
            {"contents": "   ", "parent_page": None},
            {"contents": "", "parent_page": None},
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
