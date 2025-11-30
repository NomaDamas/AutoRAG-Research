import pytest
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.exceptions import LengthMismatchError
from autorag_research.orm.schema import Chunk, Query, RetrievalRelation
from autorag_research.orm.service.text_ingestion import TextDataIngestionService


@pytest.fixture
def text_ingestion_service(session_factory: sessionmaker[Session]) -> TextDataIngestionService:
    return TextDataIngestionService(session_factory)


class TestAddQuery:
    def test_add_query_without_generation_gt(
        self, text_ingestion_service: TextDataIngestionService, db_session: Session
    ):
        query_id = text_ingestion_service.add_query("Test query without gt")

        assert query_id is not None

        # Verify by fetching from DB
        query = db_session.get(Query, query_id)
        assert query.query == "Test query without gt"
        assert query.generation_gt is None

        db_session.delete(db_session.get(Query, query_id))
        db_session.commit()

    def test_add_query_with_generation_gt(self, text_ingestion_service: TextDataIngestionService, db_session: Session):
        query_id = text_ingestion_service.add_query(
            "Test query with gt",
            generation_gt=["answer1", "answer2"],
        )

        assert query_id is not None

        # Verify by fetching from DB
        query = db_session.get(Query, query_id)
        assert query.query == "Test query with gt"
        assert query.generation_gt == ["answer1", "answer2"]

        db_session.delete(db_session.get(Query, query_id))
        db_session.commit()

    def test_add_query_with_explicit_id(self, text_ingestion_service: TextDataIngestionService, db_session: Session):
        unique_id = 900001
        query_id = text_ingestion_service.add_query(
            "Test query with explicit id",
            qid=unique_id,
        )

        assert query_id == unique_id

        # Verify by fetching from DB
        query = db_session.get(Query, query_id)
        assert query.query == "Test query with explicit id"

        db_session.delete(db_session.get(Query, unique_id))
        db_session.commit()


class TestAddQueries:
    def test_add_queries(self, text_ingestion_service: TextDataIngestionService, db_session: Session):
        queries_data = [
            ("Query 1", ["gt1"]),
            ("Query 2", None),
            ("Query 3", ["gt3a", "gt3b"]),
        ]
        query_ids = text_ingestion_service.add_queries(queries_data)

        assert len(query_ids) == 3
        assert all(qid is not None for qid in query_ids)

        # Verify by fetching from DB
        q1 = db_session.get(Query, query_ids[0])
        q2 = db_session.get(Query, query_ids[1])
        q3 = db_session.get(Query, query_ids[2])
        assert q1.query == "Query 1"
        assert q1.generation_gt == ["gt1"]
        assert q2.generation_gt is None
        assert q3.generation_gt == ["gt3a", "gt3b"]

        for qid in query_ids:
            db_session.delete(db_session.get(Query, qid))
        db_session.commit()

    def test_add_queries_with_explicit_ids(self, text_ingestion_service: TextDataIngestionService, db_session: Session):
        queries_data = [
            ("Query A", None),
            ("Query B", None),
        ]
        qids = [900011, 900012]
        query_ids = text_ingestion_service.add_queries(queries_data, qids=qids)

        assert len(query_ids) == 2
        assert query_ids[0] == 900011
        assert query_ids[1] == 900012

        for qid in query_ids:
            db_session.delete(db_session.get(Query, qid))
        db_session.commit()


class TestAddQueriesSimple:
    def test_add_queries_simple(self, text_ingestion_service: TextDataIngestionService, db_session: Session):
        query_texts = ["Simple query 1", "Simple query 2"]
        query_ids = text_ingestion_service.add_queries_simple(query_texts)

        assert len(query_ids) == 2
        assert all(qid is not None for qid in query_ids)

        # Verify by fetching from DB
        for qid in query_ids:
            q = db_session.get(Query, qid)
            assert q.generation_gt is None

        for qid in query_ids:
            db_session.delete(db_session.get(Query, qid))
        db_session.commit()

    def test_add_queries_simple_with_explicit_ids(
        self, text_ingestion_service: TextDataIngestionService, db_session: Session
    ):
        query_texts = ["Simple Q1", "Simple Q2"]
        qids = [900021, 900022]
        query_ids = text_ingestion_service.add_queries_simple(query_texts, qids=qids)

        assert query_ids[0] == 900021
        assert query_ids[1] == 900022

        for qid in query_ids:
            db_session.delete(db_session.get(Query, qid))
        db_session.commit()


class TestGetQuery:
    def test_get_query_by_text(self, text_ingestion_service: TextDataIngestionService):
        result = text_ingestion_service.get_query_by_text("What is Doc One about?")

        assert result is not None
        assert result.id == 1
        assert result.generation_gt == ["alpha"]

    def test_get_query_by_text_not_found(self, text_ingestion_service: TextDataIngestionService):
        result = text_ingestion_service.get_query_by_text("Non-existent query")

        assert result is None

    def test_get_query_by_id(self, text_ingestion_service: TextDataIngestionService):
        result = text_ingestion_service.get_query_by_id(1)

        assert result is not None
        assert result.query == "What is Doc One about?"

    def test_get_query_by_id_not_found(self, text_ingestion_service: TextDataIngestionService):
        result = text_ingestion_service.get_query_by_id(999999)

        assert result is None


class TestAddChunk:
    def test_add_chunk_standalone(self, text_ingestion_service: TextDataIngestionService, db_session: Session):
        chunk_id = text_ingestion_service.add_chunk("Test chunk content")

        assert chunk_id is not None

        # Verify by fetching from DB
        chunk = db_session.get(Chunk, chunk_id)
        assert chunk.contents == "Test chunk content"
        assert chunk.parent_caption is None

        db_session.delete(db_session.get(Chunk, chunk_id))
        db_session.commit()

    def test_add_chunk_with_parent_caption(self, text_ingestion_service: TextDataIngestionService, db_session: Session):
        chunk_id = text_ingestion_service.add_chunk("Chunk with parent", parent_caption_id=1)

        assert chunk_id is not None

        # Verify by fetching from DB
        chunk = db_session.get(Chunk, chunk_id)
        assert chunk.parent_caption == 1

        db_session.delete(db_session.get(Chunk, chunk_id))
        db_session.commit()

    def test_add_chunk_with_explicit_id(self, text_ingestion_service: TextDataIngestionService, db_session: Session):
        unique_id = 900031
        chunk_id = text_ingestion_service.add_chunk("Chunk with explicit id", chunk_id=unique_id)

        assert chunk_id == unique_id

        db_session.delete(db_session.get(Chunk, unique_id))
        db_session.commit()


class TestAddChunks:
    def test_add_chunks(self, text_ingestion_service: TextDataIngestionService, db_session: Session):
        chunks_data = [
            ("Chunk content 1", None),
            ("Chunk content 2", 1),
        ]
        chunk_ids = text_ingestion_service.add_chunks(chunks_data)

        assert len(chunk_ids) == 2
        assert all(cid is not None for cid in chunk_ids)

        # Verify by fetching from DB
        c1 = db_session.get(Chunk, chunk_ids[0])
        c2 = db_session.get(Chunk, chunk_ids[1])
        assert c1.parent_caption is None
        assert c2.parent_caption == 1

        for cid in chunk_ids:
            db_session.delete(db_session.get(Chunk, cid))
        db_session.commit()

    def test_add_chunks_with_explicit_ids(self, text_ingestion_service: TextDataIngestionService, db_session: Session):
        chunks_data = [
            ("Chunk A", None),
            ("Chunk B", None),
        ]
        explicit_ids = [900041, 900042]
        chunk_ids = text_ingestion_service.add_chunks(chunks_data, chunk_ids=explicit_ids)

        assert chunk_ids[0] == 900041
        assert chunk_ids[1] == 900042

        for cid in chunk_ids:
            db_session.delete(db_session.get(Chunk, cid))
        db_session.commit()


class TestAddChunksSimple:
    def test_add_chunks_simple(self, text_ingestion_service: TextDataIngestionService, db_session: Session):
        contents = ["Simple content 1", "Simple content 2", "Simple content 3"]
        chunk_ids = text_ingestion_service.add_chunks_simple(contents)

        assert len(chunk_ids) == 3
        assert all(cid is not None for cid in chunk_ids)

        # Verify by fetching from DB
        for i, cid in enumerate(chunk_ids):
            c = db_session.get(Chunk, cid)
            assert c.parent_caption is None
            assert c.contents == contents[i]

        for cid in chunk_ids:
            db_session.delete(db_session.get(Chunk, cid))
        db_session.commit()

    def test_add_chunks_simple_with_explicit_ids(
        self, text_ingestion_service: TextDataIngestionService, db_session: Session
    ):
        contents = ["Content X", "Content Y"]
        explicit_ids = [900051, 900052]
        chunk_ids = text_ingestion_service.add_chunks_simple(contents, chunk_ids=explicit_ids)

        assert chunk_ids[0] == 900051
        assert chunk_ids[1] == 900052

        for cid in chunk_ids:
            db_session.delete(db_session.get(Chunk, cid))
        db_session.commit()


class TestGetChunk:
    def test_get_chunk_by_id(self, text_ingestion_service: TextDataIngestionService):
        result = text_ingestion_service.get_chunk_by_id(1)

        assert result is not None
        assert result.contents == "Chunk 1-1"

    def test_get_chunk_by_id_not_found(self, text_ingestion_service: TextDataIngestionService):
        result = text_ingestion_service.get_chunk_by_id(999999)

        assert result is None

    def test_get_chunks_by_contents(self, text_ingestion_service: TextDataIngestionService):
        results = text_ingestion_service.get_chunks_by_contents("Chunk 1-1")

        assert len(results) >= 1
        assert all(c.contents == "Chunk 1-1" for c in results)


class TestAddRetrievalGT:
    def test_add_retrieval_gt_auto_increment(
        self, text_ingestion_service: TextDataIngestionService, db_session: Session
    ):
        query_id = text_ingestion_service.add_query("Test query for retrieval gt")
        chunk_id = text_ingestion_service.add_chunk("Test chunk for retrieval gt")

        relation_pk = text_ingestion_service.add_retrieval_gt(query_id, chunk_id)

        assert relation_pk == (query_id, 0, 0)

        # Verify by fetching from DB
        relation = db_session.get(RetrievalRelation, relation_pk)
        assert relation.query_id == query_id
        assert relation.chunk_id == chunk_id

        db_session.delete(db_session.get(RetrievalRelation, relation_pk))
        db_session.delete(db_session.get(Chunk, chunk_id))
        db_session.delete(db_session.get(Query, query_id))
        db_session.commit()

    def test_add_retrieval_gt_with_explicit_group(
        self, text_ingestion_service: TextDataIngestionService, db_session: Session
    ):
        query_id = text_ingestion_service.add_query("Test query explicit group")
        chunk_id = text_ingestion_service.add_chunk("Test chunk explicit group")

        relation_pk = text_ingestion_service.add_retrieval_gt(query_id, chunk_id, group_index=5, group_order=10)

        assert relation_pk == (query_id, 5, 10)

        db_session.delete(db_session.get(RetrievalRelation, relation_pk))
        db_session.delete(db_session.get(Chunk, chunk_id))
        db_session.delete(db_session.get(Query, query_id))
        db_session.commit()


class TestAddRetrievalGTSimple:
    def test_add_retrieval_gt_simple(self, text_ingestion_service: TextDataIngestionService, db_session: Session):
        query_id = text_ingestion_service.add_query("Query for simple gt")
        chunk_ids = text_ingestion_service.add_chunks_simple(["Chunk A", "Chunk B", "Chunk C"])

        relation_pks = text_ingestion_service.add_retrieval_gt_simple(query_id, chunk_ids)

        assert len(relation_pks) == 3
        assert all(pk[1] == 0 for pk in relation_pks)  # group_index == 0
        assert [pk[2] for pk in relation_pks] == [0, 1, 2]  # group_order

        # Verify by fetching from DB
        for i, pk in enumerate(relation_pks):
            r = db_session.get(RetrievalRelation, pk)
            assert r.chunk_id == chunk_ids[i]

        for pk in relation_pks:
            db_session.delete(db_session.get(RetrievalRelation, pk))
        for cid in chunk_ids:
            db_session.delete(db_session.get(Chunk, cid))
        db_session.delete(db_session.get(Query, query_id))
        db_session.commit()


class TestAddRetrievalGTMultihop:
    def test_add_retrieval_gt_multihop(self, text_ingestion_service: TextDataIngestionService, db_session: Session):
        query_id = text_ingestion_service.add_query("Query for multihop gt")
        chunk_ids = text_ingestion_service.add_chunks_simple([
            "Hop1 Chunk1",
            "Hop1 Chunk2",
            "Hop2 Chunk1",
            "Hop2 Chunk2",
        ])

        chunk_groups = [
            [chunk_ids[0], chunk_ids[1]],  # First hop
            [chunk_ids[2], chunk_ids[3]],  # Second hop
        ]
        relation_pks = text_ingestion_service.add_retrieval_gt_multihop(query_id, chunk_groups)

        assert len(relation_pks) == 4

        hop1_pks = [pk for pk in relation_pks if pk[1] == 0]  # group_index == 0
        hop2_pks = [pk for pk in relation_pks if pk[1] == 1]  # group_index == 1

        assert len(hop1_pks) == 2
        assert len(hop2_pks) == 2

        # Verify by fetching from DB
        hop1_relations = [db_session.get(RetrievalRelation, pk) for pk in hop1_pks]
        hop2_relations = [db_session.get(RetrievalRelation, pk) for pk in hop2_pks]
        assert [r.chunk_id for r in hop1_relations] == [chunk_ids[0], chunk_ids[1]]
        assert [r.chunk_id for r in hop2_relations] == [chunk_ids[2], chunk_ids[3]]

        for pk in relation_pks:
            db_session.delete(db_session.get(RetrievalRelation, pk))
        for cid in chunk_ids:
            db_session.delete(db_session.get(Chunk, cid))
        db_session.delete(db_session.get(Query, query_id))
        db_session.commit()


class TestGetRetrievalGT:
    def test_get_retrieval_gt_by_query(self, text_ingestion_service: TextDataIngestionService):
        relations = text_ingestion_service.get_retrieval_gt_by_query(1)

        assert len(relations) >= 1
        assert all(r.query_id == 1 for r in relations)


class TestEmbedding:
    def test_set_query_embedding(self, text_ingestion_service: TextDataIngestionService, db_session: Session):
        query_id = text_ingestion_service.add_query("Query to embed for test")
        embedding = [0.1] * 768

        result = text_ingestion_service.set_query_embedding(query_id, embedding)

        assert result is True

        # Verify by fetching from DB
        query = db_session.get(Query, query_id)
        assert query.embedding is not None
        assert len(list(query.embedding)) == 768

        db_session.delete(db_session.get(Query, query_id))
        db_session.commit()

    def test_set_chunk_embedding(self, text_ingestion_service: TextDataIngestionService, db_session: Session):
        chunk_id = text_ingestion_service.add_chunk("Chunk to embed for test")
        embedding = [0.2] * 768

        result = text_ingestion_service.set_chunk_embedding(chunk_id, embedding)

        assert result is True

        # Verify by fetching from DB
        chunk = db_session.get(Chunk, chunk_id)
        assert chunk.embedding is not None
        assert len(list(chunk.embedding)) == 768

        db_session.delete(db_session.get(Chunk, chunk_id))
        db_session.commit()

    def test_set_query_embedding_not_found(self, text_ingestion_service: TextDataIngestionService):
        embedding = [0.1] * 768
        result = text_ingestion_service.set_query_embedding(999999, embedding)

        assert result is False

    def test_set_chunk_embedding_not_found(self, text_ingestion_service: TextDataIngestionService):
        embedding = [0.1] * 768
        result = text_ingestion_service.set_chunk_embedding(999999, embedding)

        assert result is False

    def test_set_query_embeddings_batch(self, text_ingestion_service: TextDataIngestionService, db_session: Session):
        query_ids = text_ingestion_service.add_queries_simple(["Query 1", "Query 2", "Query 3"])
        embeddings = [[0.1 * i] * 768 for i in range(1, 4)]

        result = text_ingestion_service.set_query_embeddings(query_ids, embeddings)

        assert result == 3

        for qid in query_ids:
            db_session.delete(db_session.get(Query, qid))
        db_session.commit()

    def test_set_chunk_embeddings_batch(self, text_ingestion_service: TextDataIngestionService, db_session: Session):
        chunk_ids = text_ingestion_service.add_chunks_simple(["Chunk 1", "Chunk 2", "Chunk 3"])
        embeddings = [[0.2 * i] * 768 for i in range(1, 4)]

        result = text_ingestion_service.set_chunk_embeddings(chunk_ids, embeddings)

        assert result == 3

        for cid in chunk_ids:
            db_session.delete(db_session.get(Chunk, cid))
        db_session.commit()

    def test_set_query_embeddings_length_mismatch(self, text_ingestion_service: TextDataIngestionService):
        with pytest.raises(LengthMismatchError):
            text_ingestion_service.set_query_embeddings([1, 2], [[0.1] * 768])

    def test_set_chunk_embeddings_length_mismatch(self, text_ingestion_service: TextDataIngestionService):
        with pytest.raises(LengthMismatchError):
            text_ingestion_service.set_chunk_embeddings([1, 2], [[0.1] * 768])

    def test_set_query_embeddings_partial_success(
        self, text_ingestion_service: TextDataIngestionService, db_session: Session
    ):
        """Test that set_query_embeddings only updates existing queries."""
        query_id = text_ingestion_service.add_query("Existing query")
        embeddings = [[0.1] * 768, [0.2] * 768]

        # One valid query ID and one invalid
        result = text_ingestion_service.set_query_embeddings([query_id, 999999], embeddings)

        assert result == 1  # Only one query was updated

        db_session.delete(db_session.get(Query, query_id))
        db_session.commit()

    def test_set_chunk_embeddings_partial_success(
        self, text_ingestion_service: TextDataIngestionService, db_session: Session
    ):
        """Test that set_chunk_embeddings only updates existing chunks."""
        chunk_id = text_ingestion_service.add_chunk("Existing chunk")
        embeddings = [[0.1] * 768, [0.2] * 768]

        # One valid chunk ID and one invalid
        result = text_ingestion_service.set_chunk_embeddings([chunk_id, 999999], embeddings)

        assert result == 1  # Only one chunk was updated

        db_session.delete(db_session.get(Chunk, chunk_id))
        db_session.commit()


class TestGetStatistics:
    def test_get_statistics(self, text_ingestion_service: TextDataIngestionService):
        stats = text_ingestion_service.get_statistics()

        assert "queries" in stats
        assert "chunks" in stats
        assert "total" in stats["queries"]
        assert "total" in stats["chunks"]
        assert stats["queries"]["total"] >= 5
        assert stats["chunks"]["total"] >= 6


class TestEmbedAllQueries:
    """Tests for embed_all_queries using only existing seed data."""

    def test_embed_all_queries(self, text_ingestion_service: TextDataIngestionService, db_session: Session):
        """Test that all queries without embeddings get embedded."""
        # Count existing queries without embeddings
        initial_count = db_session.query(Query).filter(Query.embedding.is_(None)).count()
        assert initial_count > 0, "Seed data should have queries without embeddings"

        async def mock_embed_func(text: str) -> list[float]:
            return [0.1] * 768

        result = text_ingestion_service.embed_all_queries(
            embed_func=mock_embed_func,
            batch_size=2,
            max_concurrency=2,
        )

        assert result == initial_count

        # Verify all queries now have embeddings
        remaining = db_session.query(Query).filter(Query.embedding.is_(None)).count()
        assert remaining == 0

        # Cleanup: reset all query embeddings to NULL
        db_session.query(Query).update({Query.embedding: None})
        db_session.commit()


class TestEmbedAllChunks:
    def test_embed_all_chunks(self, text_ingestion_service: TextDataIngestionService, db_session: Session):
        initial_count = db_session.query(Chunk).filter(Chunk.embedding.is_(None)).count()
        assert initial_count > 0

        async def mock_embed_func(text: str) -> list[float]:
            return [0.2] * 768

        result = text_ingestion_service.embed_all_chunks(
            embed_func=mock_embed_func,
            batch_size=2,
            max_concurrency=2,
        )

        assert result == initial_count

        remaining = db_session.query(Chunk).filter(Chunk.embedding.is_(None)).count()
        assert remaining == 0

        db_session.query(Chunk).update({Chunk.embedding: None})
        db_session.commit()


class TestClean:
    def test_clean_empty_queries(self, text_ingestion_service: TextDataIngestionService, db_session: Session):
        text_ingestion_service.add_query("", qid=900101)
        text_ingestion_service.add_query("   ", qid=900102)
        text_ingestion_service.add_query("Valid query", qid=900103)

        result = text_ingestion_service.clean()

        assert result["deleted_queries"] >= 2
        assert db_session.get(Query, 900101) is None
        assert db_session.get(Query, 900102) is None
        assert db_session.get(Query, 900103) is not None

        db_session.delete(db_session.get(Query, 900103))
        db_session.commit()

    def test_clean_empty_chunks(self, text_ingestion_service: TextDataIngestionService, db_session: Session):
        text_ingestion_service.add_chunk("", chunk_id=900201)
        text_ingestion_service.add_chunk("   ", chunk_id=900202)
        text_ingestion_service.add_chunk("Valid chunk", chunk_id=900203)

        result = text_ingestion_service.clean()

        assert result["deleted_chunks"] >= 2
        assert db_session.get(Chunk, 900201) is None
        assert db_session.get(Chunk, 900202) is None
        assert db_session.get(Chunk, 900203) is not None

        db_session.delete(db_session.get(Chunk, 900203))
        db_session.commit()

    def test_clean_deletes_associated_retrieval_relations(
        self, text_ingestion_service: TextDataIngestionService, db_session: Session
    ):
        empty_query = text_ingestion_service.add_query("", qid=900301)
        valid_chunk = text_ingestion_service.add_chunk("Valid chunk for relation", chunk_id=900302)
        text_ingestion_service.add_retrieval_gt(empty_query.id, valid_chunk.id)

        valid_query = text_ingestion_service.add_query("Valid query for relation", qid=900303)
        empty_chunk = text_ingestion_service.add_chunk("", chunk_id=900304)
        text_ingestion_service.add_retrieval_gt(valid_query.id, empty_chunk.id)

        result = text_ingestion_service.clean()

        assert result["deleted_queries"] >= 1
        assert result["deleted_chunks"] >= 1
        assert db_session.get(Query, 900301) is None
        assert db_session.get(Chunk, 900304) is None
        assert db_session.get(RetrievalRelation, (900301, 0, 0)) is None
        assert db_session.get(RetrievalRelation, (900303, 0, 0)) is None

        db_session.delete(db_session.get(Chunk, 900302))
        db_session.delete(db_session.get(Query, 900303))
        db_session.commit()

    def test_clean_with_no_empty_content(self, text_ingestion_service: TextDataIngestionService, db_session: Session):
        text_ingestion_service.clean()

        assert db_session.get(Query, 1) is not None
        assert db_session.get(Chunk, 2) is not None
