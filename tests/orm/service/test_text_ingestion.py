import pytest
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.orm.schema import Chunk, Query, RetrievalRelation
from autorag_research.orm.service.text_ingestion import TextDataIngestionService


@pytest.fixture
def text_ingestion_service(session_factory: sessionmaker[Session]) -> TextDataIngestionService:
    return TextDataIngestionService(session_factory)


class TestAddQuery:
    def test_add_query_without_generation_gt(
        self, text_ingestion_service: TextDataIngestionService, db_session: Session
    ):
        query = text_ingestion_service.add_query("Test query without gt")

        assert query is not None
        assert query.id is not None
        assert query.query == "Test query without gt"
        assert query.generation_gt is None

        db_session.delete(db_session.get(Query, query.id))
        db_session.commit()

    def test_add_query_with_generation_gt(self, text_ingestion_service: TextDataIngestionService, db_session: Session):
        query = text_ingestion_service.add_query(
            "Test query with gt",
            generation_gt=["answer1", "answer2"],
        )

        assert query is not None
        assert query.id is not None
        assert query.query == "Test query with gt"
        assert query.generation_gt == ["answer1", "answer2"]

        db_session.delete(db_session.get(Query, query.id))
        db_session.commit()

    def test_add_query_with_explicit_id(self, text_ingestion_service: TextDataIngestionService, db_session: Session):
        unique_id = 900001
        query = text_ingestion_service.add_query(
            "Test query with explicit id",
            qid=unique_id,
        )

        assert query is not None
        assert query.id == unique_id
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
        queries = text_ingestion_service.add_queries(queries_data)

        assert len(queries) == 3
        assert all(q.id is not None for q in queries)
        assert queries[0].query == "Query 1"
        assert queries[0].generation_gt == ["gt1"]
        assert queries[1].generation_gt is None
        assert queries[2].generation_gt == ["gt3a", "gt3b"]

        for q in queries:
            db_session.delete(db_session.get(Query, q.id))
        db_session.commit()

    def test_add_queries_with_explicit_ids(self, text_ingestion_service: TextDataIngestionService, db_session: Session):
        queries_data = [
            ("Query A", None),
            ("Query B", None),
        ]
        qids = [900011, 900012]
        queries = text_ingestion_service.add_queries(queries_data, qids=qids)

        assert len(queries) == 2
        assert queries[0].id == 900011
        assert queries[1].id == 900012

        for q in queries:
            db_session.delete(db_session.get(Query, q.id))
        db_session.commit()


class TestAddQueriesSimple:
    def test_add_queries_simple(self, text_ingestion_service: TextDataIngestionService, db_session: Session):
        query_texts = ["Simple query 1", "Simple query 2"]
        queries = text_ingestion_service.add_queries_simple(query_texts)

        assert len(queries) == 2
        assert all(q.id is not None for q in queries)
        assert all(q.generation_gt is None for q in queries)

        for q in queries:
            db_session.delete(db_session.get(Query, q.id))
        db_session.commit()

    def test_add_queries_simple_with_explicit_ids(
        self, text_ingestion_service: TextDataIngestionService, db_session: Session
    ):
        query_texts = ["Simple Q1", "Simple Q2"]
        qids = [900021, 900022]
        queries = text_ingestion_service.add_queries_simple(query_texts, qids=qids)

        assert queries[0].id == 900021
        assert queries[1].id == 900022

        for q in queries:
            db_session.delete(db_session.get(Query, q.id))
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
        chunk = text_ingestion_service.add_chunk("Test chunk content")

        assert chunk is not None
        assert chunk.id is not None
        assert chunk.contents == "Test chunk content"
        assert chunk.parent_caption is None

        db_session.delete(db_session.get(Chunk, chunk.id))
        db_session.commit()

    def test_add_chunk_with_parent_caption(self, text_ingestion_service: TextDataIngestionService, db_session: Session):
        chunk = text_ingestion_service.add_chunk("Chunk with parent", parent_caption_id=1)

        assert chunk is not None
        assert chunk.parent_caption == 1

        db_session.delete(db_session.get(Chunk, chunk.id))
        db_session.commit()

    def test_add_chunk_with_explicit_id(self, text_ingestion_service: TextDataIngestionService, db_session: Session):
        unique_id = 900031
        chunk = text_ingestion_service.add_chunk("Chunk with explicit id", chunk_id=unique_id)

        assert chunk is not None
        assert chunk.id == unique_id

        db_session.delete(db_session.get(Chunk, unique_id))
        db_session.commit()


class TestAddChunks:
    def test_add_chunks(self, text_ingestion_service: TextDataIngestionService, db_session: Session):
        chunks_data = [
            ("Chunk content 1", None),
            ("Chunk content 2", 1),
        ]
        chunks = text_ingestion_service.add_chunks(chunks_data)

        assert len(chunks) == 2
        assert all(c.id is not None for c in chunks)
        assert chunks[0].parent_caption is None
        assert chunks[1].parent_caption == 1

        for c in chunks:
            db_session.delete(db_session.get(Chunk, c.id))
        db_session.commit()

    def test_add_chunks_with_explicit_ids(self, text_ingestion_service: TextDataIngestionService, db_session: Session):
        chunks_data = [
            ("Chunk A", None),
            ("Chunk B", None),
        ]
        chunk_ids = [900041, 900042]
        chunks = text_ingestion_service.add_chunks(chunks_data, chunk_ids=chunk_ids)

        assert chunks[0].id == 900041
        assert chunks[1].id == 900042

        for c in chunks:
            db_session.delete(db_session.get(Chunk, c.id))
        db_session.commit()


class TestAddChunksSimple:
    def test_add_chunks_simple(self, text_ingestion_service: TextDataIngestionService, db_session: Session):
        contents = ["Simple content 1", "Simple content 2", "Simple content 3"]
        chunks = text_ingestion_service.add_chunks_simple(contents)

        assert len(chunks) == 3
        assert all(c.id is not None for c in chunks)
        assert all(c.parent_caption is None for c in chunks)
        assert [c.contents for c in chunks] == contents

        for c in chunks:
            db_session.delete(db_session.get(Chunk, c.id))
        db_session.commit()

    def test_add_chunks_simple_with_explicit_ids(
        self, text_ingestion_service: TextDataIngestionService, db_session: Session
    ):
        contents = ["Content X", "Content Y"]
        chunk_ids = [900051, 900052]
        chunks = text_ingestion_service.add_chunks_simple(contents, chunk_ids=chunk_ids)

        assert chunks[0].id == 900051
        assert chunks[1].id == 900052

        for c in chunks:
            db_session.delete(db_session.get(Chunk, c.id))
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
        query = text_ingestion_service.add_query("Test query for retrieval gt")
        chunk = text_ingestion_service.add_chunk("Test chunk for retrieval gt")

        relation = text_ingestion_service.add_retrieval_gt(query.id, chunk.id)

        assert relation is not None
        assert relation.query_id == query.id
        assert relation.chunk_id == chunk.id
        assert relation.group_index == 0
        assert relation.group_order == 0

        db_session.delete(db_session.get(RetrievalRelation, (query.id, 0, 0)))
        db_session.delete(db_session.get(Chunk, chunk.id))
        db_session.delete(db_session.get(Query, query.id))
        db_session.commit()

    def test_add_retrieval_gt_with_explicit_group(
        self, text_ingestion_service: TextDataIngestionService, db_session: Session
    ):
        query = text_ingestion_service.add_query("Test query explicit group")
        chunk = text_ingestion_service.add_chunk("Test chunk explicit group")

        relation = text_ingestion_service.add_retrieval_gt(query.id, chunk.id, group_index=5, group_order=10)

        assert relation.group_index == 5
        assert relation.group_order == 10

        db_session.delete(db_session.get(RetrievalRelation, (query.id, 5, 10)))
        db_session.delete(db_session.get(Chunk, chunk.id))
        db_session.delete(db_session.get(Query, query.id))
        db_session.commit()


class TestAddRetrievalGTSimple:
    def test_add_retrieval_gt_simple(self, text_ingestion_service: TextDataIngestionService, db_session: Session):
        query = text_ingestion_service.add_query("Query for simple gt")
        chunks = text_ingestion_service.add_chunks_simple(["Chunk A", "Chunk B", "Chunk C"])

        relations = text_ingestion_service.add_retrieval_gt_simple(query.id, [c.id for c in chunks])

        assert len(relations) == 3
        assert all(r.group_index == 0 for r in relations)
        assert [r.group_order for r in relations] == [0, 1, 2]
        assert [r.chunk_id for r in relations] == [c.id for c in chunks]

        for r in relations:
            db_session.delete(db_session.get(RetrievalRelation, (r.query_id, r.group_index, r.group_order)))
        for c in chunks:
            db_session.delete(db_session.get(Chunk, c.id))
        db_session.delete(db_session.get(Query, query.id))
        db_session.commit()


class TestAddRetrievalGTMultihop:
    def test_add_retrieval_gt_multihop(self, text_ingestion_service: TextDataIngestionService, db_session: Session):
        query = text_ingestion_service.add_query("Query for multihop gt")
        chunks = text_ingestion_service.add_chunks_simple(["Hop1 Chunk1", "Hop1 Chunk2", "Hop2 Chunk1", "Hop2 Chunk2"])

        chunk_groups = [
            [chunks[0].id, chunks[1].id],  # First hop
            [chunks[2].id, chunks[3].id],  # Second hop
        ]
        relations = text_ingestion_service.add_retrieval_gt_multihop(query.id, chunk_groups)

        assert len(relations) == 4

        hop1_relations = [r for r in relations if r.group_index == 0]
        hop2_relations = [r for r in relations if r.group_index == 1]

        assert len(hop1_relations) == 2
        assert len(hop2_relations) == 2
        assert [r.chunk_id for r in hop1_relations] == [chunks[0].id, chunks[1].id]
        assert [r.chunk_id for r in hop2_relations] == [chunks[2].id, chunks[3].id]

        for r in relations:
            db_session.delete(db_session.get(RetrievalRelation, (r.query_id, r.group_index, r.group_order)))
        for c in chunks:
            db_session.delete(db_session.get(Chunk, c.id))
        db_session.delete(db_session.get(Query, query.id))
        db_session.commit()


class TestGetRetrievalGT:
    def test_get_retrieval_gt_by_query(self, text_ingestion_service: TextDataIngestionService):
        relations = text_ingestion_service.get_retrieval_gt_by_query(1)

        assert len(relations) >= 1
        assert all(r.query_id == 1 for r in relations)


class TestEmbedding:
    def test_set_query_embedding(self, text_ingestion_service: TextDataIngestionService, db_session: Session):
        query = text_ingestion_service.add_query("Query to embed for test")
        embedding = [0.1] * 768

        result = text_ingestion_service.set_query_embedding(query.id, embedding)

        assert result is not None
        assert result.embedding is not None
        assert len(list(result.embedding)) == 768

        db_session.delete(db_session.get(Query, query.id))
        db_session.commit()

    def test_set_chunk_embedding(self, text_ingestion_service: TextDataIngestionService, db_session: Session):
        chunk = text_ingestion_service.add_chunk("Chunk to embed for test")
        embedding = [0.2] * 768

        result = text_ingestion_service.set_chunk_embedding(chunk.id, embedding)

        assert result is not None
        assert result.embedding is not None
        assert len(list(result.embedding)) == 768

        db_session.delete(db_session.get(Chunk, chunk.id))
        db_session.commit()

    def test_set_query_embedding_not_found(self, text_ingestion_service: TextDataIngestionService):
        embedding = [0.1] * 768
        result = text_ingestion_service.set_query_embedding(999999, embedding)

        assert result is None

    def test_set_chunk_embedding_not_found(self, text_ingestion_service: TextDataIngestionService):
        embedding = [0.1] * 768
        result = text_ingestion_service.set_chunk_embedding(999999, embedding)

        assert result is None

    def test_set_query_embeddings_batch(self, text_ingestion_service: TextDataIngestionService, db_session: Session):
        queries = text_ingestion_service.add_queries_simple(["Query 1", "Query 2", "Query 3"])
        embeddings = [[0.1 * i] * 768 for i in range(1, 4)]

        result = text_ingestion_service.set_query_embeddings([q.id for q in queries], embeddings)

        assert result == 3

        for q in queries:
            db_session.delete(db_session.get(Query, q.id))
        db_session.commit()

    def test_set_chunk_embeddings_batch(self, text_ingestion_service: TextDataIngestionService, db_session: Session):
        chunks = text_ingestion_service.add_chunks_simple(["Chunk 1", "Chunk 2", "Chunk 3"])
        embeddings = [[0.2 * i] * 768 for i in range(1, 4)]

        result = text_ingestion_service.set_chunk_embeddings([c.id for c in chunks], embeddings)

        assert result == 3

        for c in chunks:
            db_session.delete(db_session.get(Chunk, c.id))
        db_session.commit()

    def test_set_query_embeddings_length_mismatch(self, text_ingestion_service: TextDataIngestionService):
        with pytest.raises(ValueError, match="must have the same length"):
            text_ingestion_service.set_query_embeddings([1, 2], [[0.1] * 768])

    def test_set_chunk_embeddings_length_mismatch(self, text_ingestion_service: TextDataIngestionService):
        with pytest.raises(ValueError, match="must have the same length"):
            text_ingestion_service.set_chunk_embeddings([1, 2], [[0.1] * 768])

    def test_set_query_embeddings_partial_success(
        self, text_ingestion_service: TextDataIngestionService, db_session: Session
    ):
        """Test that set_query_embeddings only updates existing queries."""
        query = text_ingestion_service.add_query("Existing query")
        embeddings = [[0.1] * 768, [0.2] * 768]

        # One valid query ID and one invalid
        result = text_ingestion_service.set_query_embeddings([query.id, 999999], embeddings)

        assert result == 1  # Only one query was updated

        db_session.delete(db_session.get(Query, query.id))
        db_session.commit()

    def test_set_chunk_embeddings_partial_success(
        self, text_ingestion_service: TextDataIngestionService, db_session: Session
    ):
        """Test that set_chunk_embeddings only updates existing chunks."""
        chunk = text_ingestion_service.add_chunk("Existing chunk")
        embeddings = [[0.1] * 768, [0.2] * 768]

        # One valid chunk ID and one invalid
        result = text_ingestion_service.set_chunk_embeddings([chunk.id, 999999], embeddings)

        assert result == 1  # Only one chunk was updated

        db_session.delete(db_session.get(Chunk, chunk.id))
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
