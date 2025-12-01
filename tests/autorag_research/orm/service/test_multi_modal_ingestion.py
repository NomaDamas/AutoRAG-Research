import pytest
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.exceptions import LengthMismatchError
from autorag_research.orm.schema import Caption, Chunk, Document, File, ImageChunk, Page, Query, RetrievalRelation
from autorag_research.orm.service.multi_modal_ingestion import MultiModalIngestionService


@pytest.fixture
def multi_modal_service(session_factory: sessionmaker[Session]) -> MultiModalIngestionService:
    return MultiModalIngestionService(session_factory)


class TestAddFiles:
    def test_add_files(self, multi_modal_service: MultiModalIngestionService, db_session: Session):
        files_data = [
            ("/test/file1.pdf", "raw"),
            ("/test/file2.jpg", "image"),
        ]
        file_ids = multi_modal_service.add_files(files_data)

        assert len(file_ids) == 2
        assert all(file_id is not None for file_id in file_ids)

        # Verify by fetching from DB
        file1 = db_session.get(File, file_ids[0])
        file2 = db_session.get(File, file_ids[1])
        assert file1.path == "/test/file1.pdf"
        assert file1.type == "raw"
        assert file2.path == "/test/file2.jpg"
        assert file2.type == "image"

        for file_id in file_ids:
            db_session.delete(db_session.get(File, file_id))
        db_session.commit()

    def test_add_files_empty_list(self, multi_modal_service: MultiModalIngestionService):
        file_ids = multi_modal_service.add_files([])

        assert file_ids == []


class TestAddDocuments:
    def test_add_documents(self, multi_modal_service: MultiModalIngestionService, db_session: Session):
        docs_data = [
            {"filename": "test1.pdf", "title": "Test Doc 1", "author": "Author1"},
            {"filename": "test2.pdf", "title": "Test Doc 2", "metadata": {"key": "value"}},
        ]
        doc_ids = multi_modal_service.add_documents(docs_data)

        assert len(doc_ids) == 2
        assert all(doc_id is not None for doc_id in doc_ids)

        # Verify by fetching from DB
        doc1 = db_session.get(Document, doc_ids[0])
        doc2 = db_session.get(Document, doc_ids[1])
        assert doc1.filename == "test1.pdf"
        assert doc1.title == "Test Doc 1"
        assert doc1.author == "Author1"
        assert doc2.doc_metadata == {"key": "value"}

        for doc_id in doc_ids:
            db_session.delete(db_session.get(Document, doc_id))
        db_session.commit()

    def test_add_documents_with_filepath(self, multi_modal_service: MultiModalIngestionService, db_session: Session):
        file_ids = multi_modal_service.add_files([("/test/doc_with_file.pdf", "raw")])
        doc_ids = multi_modal_service.add_documents([{"filename": "doc_with_file.pdf", "filepath_id": file_ids[0]}])

        doc = db_session.get(Document, doc_ids[0])
        assert doc.filepath == file_ids[0]

        db_session.delete(db_session.get(Document, doc_ids[0]))
        db_session.delete(db_session.get(File, file_ids[0]))
        db_session.commit()


class TestAddPages:
    def test_add_pages(self, multi_modal_service: MultiModalIngestionService, db_session: Session):
        pages_data = [
            {"document_id": 1, "page_num": 100, "metadata": {"dpi": 150}},
            {"document_id": 1, "page_num": 101},
        ]
        page_ids = multi_modal_service.add_pages(pages_data)

        assert len(page_ids) == 2
        assert all(page_id is not None for page_id in page_ids)

        # Verify by fetching from DB
        page1 = db_session.get(Page, page_ids[0])
        assert page1.document_id == 1
        assert page1.page_num == 100
        assert page1.page_metadata == {"dpi": 150}

        for page_id in page_ids:
            db_session.delete(db_session.get(Page, page_id))
        db_session.commit()

    def test_add_pages_with_image_content(self, multi_modal_service: MultiModalIngestionService, db_session: Session):
        image_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        page_ids = multi_modal_service.add_pages([
            {"document_id": 1, "page_num": 200, "image_content": image_bytes, "mimetype": "image/png"}
        ])

        page = db_session.get(Page, page_ids[0])
        assert page.image_content == image_bytes
        assert page.mimetype == "image/png"

        db_session.delete(db_session.get(Page, page_ids[0]))
        db_session.commit()


class TestAddCaptions:
    def test_add_captions(self, multi_modal_service: MultiModalIngestionService, db_session: Session):
        captions_data = [
            (1, "Test caption 1 for page"),
            (2, "Test caption 2 for page"),
        ]
        caption_ids = multi_modal_service.add_captions(captions_data)

        assert len(caption_ids) == 2
        assert all(caption_id is not None for caption_id in caption_ids)

        # Verify by fetching from DB
        caption1 = db_session.get(Caption, caption_ids[0])
        assert caption1.page_id == 1
        assert caption1.contents == "Test caption 1 for page"

        for caption_id in caption_ids:
            db_session.delete(db_session.get(Caption, caption_id))
        db_session.commit()


class TestAddChunks:
    def test_add_chunks(self, multi_modal_service: MultiModalIngestionService, db_session: Session):
        chunks_data = [
            ("Test chunk content 1", None),
            ("Test chunk content 2", 1),
        ]
        chunk_ids = multi_modal_service.add_chunks(chunks_data)

        assert len(chunk_ids) == 2
        assert all(chunk_id is not None for chunk_id in chunk_ids)

        # Verify by fetching from DB
        chunk1 = db_session.get(Chunk, chunk_ids[0])
        chunk2 = db_session.get(Chunk, chunk_ids[1])
        assert chunk1.contents == "Test chunk content 1"
        assert chunk1.parent_caption is None
        assert chunk2.parent_caption == 1

        for chunk_id in chunk_ids:
            db_session.delete(db_session.get(Chunk, chunk_id))
        db_session.commit()


class TestAddImageChunks:
    def test_add_image_chunks(self, multi_modal_service: MultiModalIngestionService, db_session: Session):
        image_bytes1 = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        image_bytes2 = b"\xff\xd8\xff\xe0\x00\x10JFIF"

        image_chunks_data = [
            (image_bytes1, "image/png", 1),
            (image_bytes2, "image/jpeg", None),
        ]
        image_chunk_ids = multi_modal_service.add_image_chunks(image_chunks_data)

        assert len(image_chunk_ids) == 2
        assert all(ic_id is not None for ic_id in image_chunk_ids)

        # Verify by fetching from DB
        ic1 = db_session.get(ImageChunk, image_chunk_ids[0])
        ic2 = db_session.get(ImageChunk, image_chunk_ids[1])
        assert ic1.content == image_bytes1
        assert ic1.mimetype == "image/png"
        assert ic1.parent_page == 1
        assert ic2.content == image_bytes2
        assert ic2.mimetype == "image/jpeg"
        assert ic2.parent_page is None

        for ic_id in image_chunk_ids:
            db_session.delete(db_session.get(ImageChunk, ic_id))
        db_session.commit()


class TestAddQueries:
    def test_add_queries(self, multi_modal_service: MultiModalIngestionService, db_session: Session):
        queries_data = [
            ("Test query 1", ["answer1", "answer2"]),
            ("Test query 2", None),
        ]
        query_ids = multi_modal_service.add_queries(queries_data)

        assert len(query_ids) == 2
        assert all(query_id is not None for query_id in query_ids)

        # Verify by fetching from DB
        query1 = db_session.get(Query, query_ids[0])
        query2 = db_session.get(Query, query_ids[1])
        assert query1.query == "Test query 1"
        assert query1.generation_gt == ["answer1", "answer2"]
        assert query2.generation_gt is None

        for query_id in query_ids:
            db_session.delete(db_session.get(Query, query_id))
        db_session.commit()


class TestAddRetrievalGTBatch:
    def test_add_retrieval_gt_batch_with_chunks(
        self, multi_modal_service: MultiModalIngestionService, db_session: Session
    ):
        query_ids = multi_modal_service.add_queries([("Query for batch gt", None)])
        chunk_ids = multi_modal_service.add_chunks([("Chunk for batch gt", None)])

        relations_data = [
            {"query_id": query_ids[0], "chunk_id": chunk_ids[0], "group_index": 0, "group_order": 0},
        ]
        relation_pks = multi_modal_service.add_retrieval_gt_batch(relations_data)

        assert len(relation_pks) == 1
        assert relation_pks[0] == (query_ids[0], 0, 0)

        # Verify by fetching from DB
        relation = db_session.get(RetrievalRelation, relation_pks[0])
        assert relation.chunk_id == chunk_ids[0]
        assert relation.image_chunk_id is None

        db_session.delete(db_session.get(RetrievalRelation, relation_pks[0]))
        db_session.delete(db_session.get(Chunk, chunk_ids[0]))
        db_session.delete(db_session.get(Query, query_ids[0]))
        db_session.commit()

    def test_add_retrieval_gt_batch_with_image_chunks(
        self, multi_modal_service: MultiModalIngestionService, db_session: Session
    ):
        query_ids = multi_modal_service.add_queries([("Query for image batch gt", None)])
        image_bytes = b"\x89PNG\r\n\x1a\n"
        image_chunk_ids = multi_modal_service.add_image_chunks([(image_bytes, "image/png", None)])

        relations_data = [
            {"query_id": query_ids[0], "image_chunk_id": image_chunk_ids[0], "group_index": 0, "group_order": 0},
        ]
        relation_pks = multi_modal_service.add_retrieval_gt_batch(relations_data)

        assert len(relation_pks) == 1
        assert relation_pks[0] == (query_ids[0], 0, 0)

        # Verify by fetching from DB
        relation = db_session.get(RetrievalRelation, relation_pks[0])
        assert relation.chunk_id is None
        assert relation.image_chunk_id == image_chunk_ids[0]

        db_session.delete(db_session.get(RetrievalRelation, relation_pks[0]))
        db_session.delete(db_session.get(ImageChunk, image_chunk_ids[0]))
        db_session.delete(db_session.get(Query, query_ids[0]))
        db_session.commit()

    def test_add_retrieval_gt_batch_validation_error_both_none(
        self, multi_modal_service: MultiModalIngestionService, db_session: Session
    ):
        query_ids = multi_modal_service.add_queries([("Query for validation", None)])

        with pytest.raises(ValueError, match="Exactly one of chunk_id or image_chunk_id"):
            multi_modal_service.add_retrieval_gt_batch([{"query_id": query_ids[0], "group_index": 0, "group_order": 0}])

        db_session.delete(db_session.get(Query, query_ids[0]))
        db_session.commit()

    def test_add_retrieval_gt_batch_validation_error_both_set(
        self, multi_modal_service: MultiModalIngestionService, db_session: Session
    ):
        query_ids = multi_modal_service.add_queries([("Query for validation both", None)])

        with pytest.raises(ValueError, match="Exactly one of chunk_id or image_chunk_id"):
            multi_modal_service.add_retrieval_gt_batch([
                {"query_id": query_ids[0], "chunk_id": 1, "image_chunk_id": 1, "group_index": 0, "group_order": 0}
            ])

        db_session.delete(db_session.get(Query, query_ids[0]))
        db_session.commit()


class TestAddRetrievalGTMultihopMixed:
    def test_add_retrieval_gt_multihop_mixed(
        self, multi_modal_service: MultiModalIngestionService, db_session: Session
    ):
        query_ids = multi_modal_service.add_queries([("Query for mixed multihop", None)])
        chunk_ids = multi_modal_service.add_chunks([("Mixed hop chunk 1", None), ("Mixed hop chunk 2", None)])
        image_bytes = b"\x89PNG\r\n\x1a\n"
        image_chunk_ids = multi_modal_service.add_image_chunks([(image_bytes, "image/png", None)])

        groups = [
            [("chunk", chunk_ids[0]), ("chunk", chunk_ids[1])],
            [("image_chunk", image_chunk_ids[0])],
        ]
        relation_pks = multi_modal_service.add_retrieval_gt_multihop(query_ids[0], groups)

        assert len(relation_pks) == 3

        hop0_pks = [pk for pk in relation_pks if pk[1] == 0]  # group_index == 0
        hop1_pks = [pk for pk in relation_pks if pk[1] == 1]  # group_index == 1

        assert len(hop0_pks) == 2
        assert len(hop1_pks) == 1

        # Verify by fetching from DB
        for pk in hop0_pks:
            relation = db_session.get(RetrievalRelation, pk)
            assert relation.chunk_id is not None
            assert relation.image_chunk_id is None

        hop1_relation = db_session.get(RetrievalRelation, hop1_pks[0])
        assert hop1_relation.image_chunk_id == image_chunk_ids[0]
        assert hop1_relation.chunk_id is None

        for pk in relation_pks:
            db_session.delete(db_session.get(RetrievalRelation, pk))
        for chunk_id in chunk_ids:
            db_session.delete(db_session.get(Chunk, chunk_id))
        db_session.delete(db_session.get(ImageChunk, image_chunk_ids[0]))
        db_session.delete(db_session.get(Query, query_ids[0]))
        db_session.commit()

    def test_add_retrieval_gt_multihop_mixed_invalid_type(
        self, multi_modal_service: MultiModalIngestionService, db_session: Session
    ):
        query_ids = multi_modal_service.add_queries([("Query for invalid type", None)])

        with pytest.raises(ValueError, match="Invalid item type"):
            multi_modal_service.add_retrieval_gt_multihop(query_ids[0], [[("invalid_type", 1)]])

        db_session.delete(db_session.get(Query, query_ids[0]))
        db_session.commit()


class TestSetEmbeddings:
    def test_set_query_embeddings(self, multi_modal_service: MultiModalIngestionService, db_session: Session):
        query_ids = multi_modal_service.add_queries([("Query to embed 1", None), ("Query to embed 2", None)])
        embeddings = [[0.1] * 768, [0.2] * 768]

        result = multi_modal_service.set_query_embeddings(query_ids, embeddings)

        assert result == 2

        for query_id in query_ids:
            db_session.delete(db_session.get(Query, query_id))
        db_session.commit()

    def test_set_chunk_embeddings(self, multi_modal_service: MultiModalIngestionService, db_session: Session):
        chunk_ids = multi_modal_service.add_chunks([("Chunk to embed 1", None), ("Chunk to embed 2", None)])
        embeddings = [[0.1] * 768, [0.2] * 768]

        result = multi_modal_service.set_chunk_embeddings(chunk_ids, embeddings)

        assert result == 2

        for chunk_id in chunk_ids:
            db_session.delete(db_session.get(Chunk, chunk_id))
        db_session.commit()

    def test_set_image_chunk_embeddings(self, multi_modal_service: MultiModalIngestionService, db_session: Session):
        image_bytes1 = b"\x89PNG\r\n\x1a\n"
        image_bytes2 = b"\xff\xd8\xff\xe0"
        image_chunk_ids = multi_modal_service.add_image_chunks([
            (image_bytes1, "image/png", None),
            (image_bytes2, "image/jpeg", None),
        ])
        embeddings = [[0.1] * 768, [0.2] * 768]

        result = multi_modal_service.set_image_chunk_embeddings(image_chunk_ids, embeddings)

        assert result == 2

        for ic_id in image_chunk_ids:
            db_session.delete(db_session.get(ImageChunk, ic_id))
        db_session.commit()

    def test_set_query_embeddings_length_mismatch(self, multi_modal_service: MultiModalIngestionService):
        with pytest.raises(LengthMismatchError):
            multi_modal_service.set_query_embeddings([1, 2], [[0.1] * 768])

    def test_set_chunk_embeddings_length_mismatch(self, multi_modal_service: MultiModalIngestionService):
        with pytest.raises(LengthMismatchError):
            multi_modal_service.set_chunk_embeddings([1, 2], [[0.1] * 768])

    def test_set_image_chunk_embeddings_length_mismatch(self, multi_modal_service: MultiModalIngestionService):
        with pytest.raises(LengthMismatchError):
            multi_modal_service.set_image_chunk_embeddings([1, 2], [[0.1] * 768])

    def test_set_query_embeddings_partial_success(
        self, multi_modal_service: MultiModalIngestionService, db_session: Session
    ):
        query_ids = multi_modal_service.add_queries([("Existing query for partial", None)])
        embeddings = [[0.1] * 768, [0.2] * 768]

        result = multi_modal_service.set_query_embeddings([query_ids[0], 999999], embeddings)

        assert result == 1

        db_session.delete(db_session.get(Query, query_ids[0]))
        db_session.commit()


class TestGetStatistics:
    def test_get_statistics(self, multi_modal_service: MultiModalIngestionService):
        stats = multi_modal_service.get_statistics()

        assert "files" in stats
        assert "documents" in stats
        assert "pages" in stats
        assert "captions" in stats
        assert "chunks" in stats
        assert "image_chunks" in stats
        assert "queries" in stats
        assert "retrieval_relations" in stats

        assert stats["files"] >= 10
        assert stats["documents"] >= 5
        assert stats["pages"] >= 10
        assert stats["captions"] >= 10
        assert stats["chunks"]["total"] >= 6
        assert stats["image_chunks"]["total"] >= 5

        assert "with_embeddings" in stats["chunks"]
        assert "without_embeddings" in stats["chunks"]
        assert "with_embeddings" in stats["image_chunks"]
        assert "without_embeddings" in stats["image_chunks"]


@pytest.mark.xdist_group("embed_tests")
class TestEmbedAllQueries:
    def test_embed_all_queries(self, multi_modal_service: MultiModalIngestionService, db_session: Session):
        initial_count = db_session.query(Query).filter(Query.embedding.is_(None)).count()
        assert initial_count > 0

        async def mock_embed_func(text: str) -> list[float]:
            return [0.1] * 768

        result = multi_modal_service.embed_all_queries(embed_func=mock_embed_func, batch_size=2, max_concurrency=2)

        assert result == initial_count

        remaining = db_session.query(Query).filter(Query.embedding.is_(None)).count()
        assert remaining == 0

        db_session.query(Query).update({Query.embedding: None})
        db_session.commit()


@pytest.mark.xdist_group("embed_tests")
class TestEmbedAllChunks:
    def test_embed_all_chunks(self, multi_modal_service: MultiModalIngestionService, db_session: Session):
        initial_count = db_session.query(Chunk).filter(Chunk.embedding.is_(None)).count()
        assert initial_count > 0

        async def mock_embed_func(text: str) -> list[float]:
            return [0.2] * 768

        result = multi_modal_service.embed_all_chunks(embed_func=mock_embed_func, batch_size=2, max_concurrency=2)

        assert result == initial_count

        remaining = db_session.query(Chunk).filter(Chunk.embedding.is_(None)).count()
        assert remaining == 0

        db_session.query(Chunk).update({Chunk.embedding: None})
        db_session.commit()


@pytest.mark.xdist_group("embed_tests")
class TestEmbedAllImageChunks:
    def test_embed_all_image_chunks(self, multi_modal_service: MultiModalIngestionService, db_session: Session):
        initial_count = db_session.query(ImageChunk).filter(ImageChunk.embedding.is_(None)).count()
        assert initial_count > 0

        async def mock_embed_func(image_content: bytes) -> list[float]:
            return [0.3] * 768

        result = multi_modal_service.embed_all_image_chunks(embed_func=mock_embed_func, batch_size=2, max_concurrency=2)

        assert result == initial_count

        remaining = db_session.query(ImageChunk).filter(ImageChunk.embedding.is_(None)).count()
        assert remaining == 0

        db_session.query(ImageChunk).update({ImageChunk.embedding: None})
        db_session.commit()


class TestCustomSchema:
    def test_with_custom_schema(self, session_factory: sessionmaker[Session], db_session: Session):
        from autorag_research.orm.schema_factory import create_schema

        custom_schema = create_schema(embedding_dim=1024)
        service = MultiModalIngestionService(session_factory, schema=custom_schema)

        stats = service.get_statistics()

        assert stats["files"] >= 10
