import pytest

from autorag_research.orm.service.multi_modal_ingestion import MultiModalIngestionService


class TestMultiModalIngestionService:
    @pytest.fixture
    def service(self, session_factory):
        return MultiModalIngestionService(session_factory)

    def test_add_files(self, service):
        files = [
            {"path": "/test/file1.pdf", "type": "raw"},
            {"path": "/test/file2.png", "type": "image"},
        ]
        ids = service.add_files(files)
        assert len(ids) == 2
        assert all(isinstance(id_, int) for id_ in ids)

        with service._create_uow() as uow:
            for id_ in ids:
                uow.files.delete_by_id(id_)
            uow.commit()

    def test_add_documents(self, service):
        documents = [
            {"filename": "test_doc.pdf", "title": "Test", "author": "test", "path": 1, "metadata": None},
        ]
        ids = service.add_documents(documents)
        assert len(ids) == 1

        with service._create_uow() as uow:
            for id_ in ids:
                uow.documents.delete_by_id(id_)
            uow.commit()

    def test_add_pages(self, service):
        pages = [
            {"document_id": 1, "page_num": 99, "image_contents": None, "mimetype": None, "metadata": None},
        ]
        ids = service.add_pages(pages)
        assert len(ids) == 1

        with service._create_uow() as uow:
            for id_ in ids:
                uow.pages.delete_by_id(id_)
            uow.commit()

    def test_add_captions(self, service):
        captions = [
            {"page_id": 1, "contents": "Test caption content"},
        ]
        ids = service.add_captions(captions)
        assert len(ids) == 1

        with service._create_uow() as uow:
            for id_ in ids:
                uow.captions.delete_by_id(id_)
            uow.commit()

    def test_add_image_chunks(self, service):
        image_chunks = [
            {"contents": b"\x00\x01\x02", "mimetype": "image/png", "parent_page": 1},
        ]
        ids = service.add_image_chunks(image_chunks)
        assert len(ids) == 1

        with service._create_uow() as uow:
            for id_ in ids:
                uow.image_chunks.delete_by_id(id_)
            uow.commit()

    def test_set_image_chunk_embeddings(self, service):
        image_chunk_ids = [1, 2]
        embeddings = [[0.5] * 768, [0.6] * 768]
        updated = service.set_image_chunk_embeddings(image_chunk_ids, embeddings)
        assert updated == 2

        with service._create_uow() as uow:
            for icid in image_chunk_ids:
                image_chunk = uow.image_chunks.get_by_id(icid)
                image_chunk.embedding = None
            uow.commit()

    def test_set_image_chunk_multi_embeddings(self, service):
        image_chunk_ids = [1, 2]
        multi_embeddings = [[[0.5] * 768] * 3, [[0.6] * 768] * 3]
        updated = service.set_image_chunk_multi_embeddings(image_chunk_ids, multi_embeddings)
        assert updated == 2

        with service._create_uow() as uow:
            for icid in image_chunk_ids:
                image_chunk = uow.image_chunks.get_by_id(icid)
                image_chunk.embeddings = None
            uow.commit()

    def test_get_statistics(self, service):
        stats = service.get_statistics()
        assert "files" in stats
        assert "documents" in stats
        assert "pages" in stats
        assert "captions" in stats
        assert "chunks" in stats
        assert "image_chunks" in stats
        assert "queries" in stats
        assert stats["files"] >= 10
        assert stats["documents"] >= 5
