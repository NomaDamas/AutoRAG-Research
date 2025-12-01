import pytest
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.orm.schema import Chunk, Query, RetrievalRelation
from autorag_research.orm.service.text_ingestion import TextDataIngestionService


@pytest.fixture
def text_ingestion_service(session_factory: sessionmaker[Session]) -> TextDataIngestionService:
    return TextDataIngestionService(session_factory)


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
        text_ingestion_service.add_retrieval_gt(empty_query, valid_chunk)

        valid_query = text_ingestion_service.add_query("Valid query for relation", qid=900303)
        empty_chunk = text_ingestion_service.add_chunk("", chunk_id=900304)
        text_ingestion_service.add_retrieval_gt(valid_query, empty_chunk)

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
