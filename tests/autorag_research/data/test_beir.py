import os

import pytest
from llama_index.core import MockEmbedding
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from autorag_research.data.beir import BEIRIngestor, _reassign_string_ids_to_integers
from autorag_research.orm.schema_factory import create_schema
from autorag_research.orm.service.text_ingestion import TextDataIngestionService
from autorag_research.orm.util import create_database, drop_database, install_vector_extensions

EMBEDDING_DIM = 768


@pytest.fixture(scope="session")
def beir_db_engine():
    host = os.getenv("POSTGRES_HOST", "localhost")
    user = os.getenv("POSTGRES_USER", "postgres")
    pwd = os.getenv("POSTGRES_PASSWORD", "postgres")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    db_name = "autorag_research_beir_test"

    schema = create_schema(EMBEDDING_DIM)

    create_database(host, user, pwd, db_name, port=port)
    install_vector_extensions(host, user, pwd, db_name, port=port)
    url = f"postgresql+psycopg://{user}:{pwd}@{host}:{port}/autorag_research_beir_test"

    engine = create_engine(
        url,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
    )
    schema.Base.metadata.create_all(engine)
    yield engine
    engine.dispose()

    drop_database(host, user, pwd, db_name, port=port)


@pytest.fixture(scope="session")
def session_factory_beir(beir_db_engine):
    return sessionmaker(bind=beir_db_engine)


@pytest.fixture
def service(session_factory_beir):
    service = TextDataIngestionService(session_factory_beir)
    yield service


@pytest.fixture
def beir_ingestor(db_session, service):
    ingestor = BEIRIngestor(service, MockEmbedding(EMBEDDING_DIM), "scifact")
    yield ingestor


@pytest.mark.data
def test_beir_ingest_embed_all(beir_ingestor):
    beir_ingestor.ingest(subset="test")
    stats = beir_ingestor.service.get_statistics()
    assert stats["queries"]["total"] == 300
    assert stats["chunks"]["total"] == 5183
    assert stats["chunks"]["with_embeddings"] == 0

    beir_ingestor.embed_all(max_concurrency=16)
    stats = beir_ingestor.service.get_statistics()
    assert stats["queries"]["total"] == 300
    assert stats["chunks"]["total"] == 5183
    assert stats["chunks"]["with_embeddings"] == 5183
    assert stats["chunks"]["without_embeddings"] == 0


class TestReassignStringIdsToIntegers:
    """Tests for _reassign_string_ids_to_integers function."""

    def test_already_valid_integer_ids(self):
        """Test that valid integer string IDs are preserved."""
        corpus = {"1": {"title": "Doc 1", "text": "Content 1"}, "2": {"title": "Doc 2", "text": "Content 2"}}
        queries = {"10": "Query A", "20": "Query B"}
        qrels = {"10": {"1": 1, "2": 0}, "20": {"2": 1}}

        new_corpus, new_queries, new_qrels, query_id_map, corpus_id_map = _reassign_string_ids_to_integers(
            corpus, queries, qrels
        )

        # Should keep original IDs
        assert new_corpus == corpus
        assert new_queries == queries
        assert new_qrels == qrels
        # Mappings should be identity mappings
        assert query_id_map == {"10": 10, "20": 20}
        assert corpus_id_map == {"1": 1, "2": 2}

    def test_non_numeric_string_ids(self):
        """Test reassignment when IDs are non-numeric strings."""
        corpus = {"doc_a": {"title": "Doc A", "text": "Content A"}, "doc_b": {"title": "Doc B", "text": "Content B"}}
        queries = {"q_1": "Query 1", "q_2": "Query 2"}
        qrels = {"q_1": {"doc_a": 1, "doc_b": 0}, "q_2": {"doc_b": 1}}

        new_corpus, new_queries, new_qrels, query_id_map, corpus_id_map = _reassign_string_ids_to_integers(
            corpus, queries, qrels
        )

        # Check that new IDs are numeric strings
        assert all(k.isdigit() for k in new_corpus)
        assert all(k.isdigit() for k in new_queries)
        assert all(k.isdigit() for k in new_qrels)

        # Check mappings are created correctly
        assert query_id_map == {"q_1": 1, "q_2": 2}
        assert corpus_id_map == {"doc_a": 1, "doc_b": 2}

        # Check content is preserved
        assert new_corpus["1"] == {"title": "Doc A", "text": "Content A"}
        assert new_corpus["2"] == {"title": "Doc B", "text": "Content B"}
        assert new_queries["1"] == "Query 1"
        assert new_queries["2"] == "Query 2"

        # Check qrels are updated correctly
        assert new_qrels["1"] == {"1": 1, "2": 0}
        assert new_qrels["2"] == {"2": 1}

    def test_duplicate_ids_when_converted(self):
        """Test reassignment when IDs would collide after conversion (e.g., '1' and '01')."""
        corpus = {"1": {"title": "Doc 1", "text": "Content 1"}, "01": {"title": "Doc 01", "text": "Content 01"}}
        queries = {"10": "Query 10", "010": "Query 010"}
        qrels = {"10": {"1": 1}, "010": {"01": 1}}

        new_corpus, new_queries, _, query_id_map, corpus_id_map = _reassign_string_ids_to_integers(
            corpus, queries, qrels
        )

        # Should reassign to avoid collision
        assert len(new_corpus) == 2
        assert len(new_queries) == 2
        assert "1" in new_corpus and "2" in new_corpus
        assert "1" in new_queries and "2" in new_queries

        # Verify mappings distinguish the original IDs
        assert query_id_map["10"] != query_id_map["010"]
        assert corpus_id_map["1"] != corpus_id_map["01"]

    def test_only_queries_need_reassignment(self):
        """Test when only query IDs need reassignment."""
        corpus = {"1": {"title": "Doc 1", "text": "Content 1"}, "2": {"title": "Doc 2", "text": "Content 2"}}
        queries = {"query_a": "Query A", "query_b": "Query B"}
        qrels = {"query_a": {"1": 1}, "query_b": {"2": 1}}

        _, new_queries, new_qrels, query_id_map, corpus_id_map = _reassign_string_ids_to_integers(
            corpus, queries, qrels
        )

        # Corpus should be unchanged, queries should be reassigned
        assert corpus_id_map == {"1": 1, "2": 2}
        assert query_id_map == {"query_a": 1, "query_b": 2}
        assert new_queries["1"] == "Query A"
        assert new_qrels["1"]["1"] == 1

    def test_only_corpus_needs_reassignment(self):
        """Test when only corpus IDs need reassignment."""
        corpus = {"doc_x": {"title": "Doc X", "text": "Content X"}, "doc_y": {"title": "Doc Y", "text": "Content Y"}}
        queries = {"1": "Query 1", "2": "Query 2"}
        qrels = {"1": {"doc_x": 1}, "2": {"doc_y": 1}}

        new_corpus, _, new_qrels, query_id_map, corpus_id_map = _reassign_string_ids_to_integers(corpus, queries, qrels)

        # Queries should be unchanged, corpus should be reassigned
        assert query_id_map == {"1": 1, "2": 2}
        assert corpus_id_map == {"doc_x": 1, "doc_y": 2}
        assert new_corpus["1"] == {"title": "Doc X", "text": "Content X"}
        assert new_qrels["1"]["1"] == 1

    def test_qrels_with_missing_corpus_ids(self):
        """Test that qrels referencing non-existent corpus IDs are skipped."""
        corpus = {"doc_a": {"title": "Doc A", "text": "Content A"}}
        queries = {"q_1": "Query 1"}
        qrels = {"q_1": {"doc_a": 1, "doc_missing": 1}}

        _, _, new_qrels, query_id_map, corpus_id_map = _reassign_string_ids_to_integers(corpus, queries, qrels)

        # The missing corpus ID should be skipped in qrels
        assert "doc_missing" not in corpus_id_map
        new_qid = str(query_id_map["q_1"])
        assert len(new_qrels[new_qid]) == 1
        assert str(corpus_id_map["doc_a"]) in new_qrels[new_qid]

    def test_empty_inputs(self):
        """Test handling of empty dictionaries."""
        corpus = {}
        queries = {}
        qrels = {}

        new_corpus, new_queries, new_qrels, query_id_map, corpus_id_map = _reassign_string_ids_to_integers(
            corpus, queries, qrels
        )

        assert new_corpus == {}
        assert new_queries == {}
        assert new_qrels == {}
        assert query_id_map == {}
        assert corpus_id_map == {}

    def test_preserves_relevance_scores(self):
        """Test that relevance scores in qrels are preserved."""
        corpus = {"a": {"title": "A", "text": "A"}, "b": {"title": "B", "text": "B"}}
        queries = {"x": "Query X"}
        qrels = {"x": {"a": 2, "b": 1}}

        _, _, new_qrels, query_id_map, corpus_id_map = _reassign_string_ids_to_integers(corpus, queries, qrels)

        new_qid = str(query_id_map["x"])
        new_cid_a = str(corpus_id_map["a"])
        new_cid_b = str(corpus_id_map["b"])

        assert new_qrels[new_qid][new_cid_a] == 2
        assert new_qrels[new_qid][new_cid_b] == 1
