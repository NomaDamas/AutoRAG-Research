"""Unit tests for BRIGHTIngestor.

Tests cover:
- Constructor validation (domain, document_mode)
- Domain selection logic
- Document mode selection
- ID prefixing/transformation
- Corpus, query, and relation ingestion flows
- Edge cases and error handling
"""

from unittest.mock import patch

import pytest
from llama_index.core import MockEmbedding

from autorag_research.data.bright import (
    BRIGHT_DOMAINS,
    DOMAINS_WITH_LONG_DOCS,
    DOMAINS_WITHOUT_LONG_DOCS,
    BRIGHTIngestor,
    _get_gold_ids,
    _make_chunk_id,
    _make_query_id,
)
from autorag_research.exceptions import ServiceNotSetError
from autorag_research.orm.schema_factory import create_schema
from autorag_research.orm.service.text_ingestion import TextDataIngestionService
from autorag_research.orm.util import create_database, drop_database, install_vector_extensions

EMBEDDING_DIM = 768
BRIGHT_TEST_DB_NAME = "bright_ingestor_test"


# ==================== Fixtures ====================


@pytest.fixture
def mock_embedding_model():
    return MockEmbedding(EMBEDDING_DIM)


@pytest.fixture(scope="function")
def bright_test_db():
    """Create a separate database for BRIGHT ingestor tests with string PKs."""
    import os

    from sqlalchemy import create_engine
    from sqlalchemy.orm import scoped_session, sessionmaker

    host = os.getenv("POSTGRES_HOST", "localhost")
    user = os.getenv("POSTGRES_USER", "postgres")
    pwd = os.getenv("POSTGRES_PASSWORD", "postgres")
    port = int(os.getenv("POSTGRES_PORT", "5432"))

    # Create a new database for BRIGHT tests
    create_database(host=host, user=user, password=pwd, database=BRIGHT_TEST_DB_NAME, port=port)
    install_vector_extensions(host=host, user=user, password=pwd, database=BRIGHT_TEST_DB_NAME, port=port)

    # Create engine and session for the new database
    postgres_url = f"postgresql+psycopg://{user}:{pwd}@{host}:{port}/{BRIGHT_TEST_DB_NAME}"
    engine = create_engine(postgres_url, pool_pre_ping=True)

    # Create schema with string primary keys
    schema = create_schema(embedding_dim=EMBEDDING_DIM, primary_key_type="string")
    schema.Base.metadata.create_all(engine)

    # Create session factory
    session_factory = scoped_session(sessionmaker(bind=engine))

    yield {"schema": schema, "engine": engine, "session_factory": session_factory}

    # Cleanup
    session_factory.remove()
    engine.dispose()
    drop_database(host=host, user=user, password=pwd, database=BRIGHT_TEST_DB_NAME, port=port, force=True)


@pytest.fixture
def text_service(bright_test_db):
    """Create TextDataIngestionService with string primary key schema."""
    return TextDataIngestionService(bright_test_db["session_factory"], schema=bright_test_db["schema"])


@pytest.fixture
def bright_db_session(bright_test_db):
    """Create a database session for BRIGHT tests."""
    session = bright_test_db["session_factory"]()
    yield session
    session.rollback()
    bright_test_db["session_factory"].remove()


@pytest.fixture
def sample_corpus_data():
    return [
        {"id": "doc_001", "content": "This is the first document content."},
        {"id": "doc_002", "content": "This is the second document content."},
        {"id": "doc_003", "content": "This is the third document content."},
    ]


@pytest.fixture
def sample_long_corpus_data():
    return [
        {"id": "long_doc_001", "content": "This is the first long document content."},
        {"id": "long_doc_002", "content": "This is the second long document content."},
    ]


@pytest.fixture
def sample_examples_data():
    return [
        {
            "id": "q_001",
            "query": "What is the first document about?",
            "gold_ids": ["doc_001", "doc_002"],
            "gold_ids_long": ["long_doc_001"],
            "gold_answer": "The first document is about testing.",
            "reasoning": "Based on the content provided.",
            "excluded_ids": ["doc_003"],
        },
        {
            "id": "q_002",
            "query": "What is the second document about?",
            "gold_ids": ["doc_002"],
            "gold_ids_long": ["long_doc_002", "N/A"],
            "gold_answer": "N/A",
            "reasoning": "empty",
            "excluded_ids": ["N/A"],
        },
    ]


def create_mock_dataset(data: list[dict]):
    yield from data


# ==================== Constructor Validation Tests ====================


class TestBRIGHTIngestorInit:
    def test_init_default_domains(self, mock_embedding_model):
        ingestor = BRIGHTIngestor(mock_embedding_model)
        assert ingestor.domains == BRIGHT_DOMAINS
        assert ingestor.document_mode == "short"

    def test_init_custom_domains(self, mock_embedding_model):
        custom_domains = ["biology", "economics"]
        ingestor = BRIGHTIngestor(mock_embedding_model, domains=custom_domains)
        assert ingestor.domains == custom_domains

    def test_init_custom_document_mode_long(self, mock_embedding_model):
        ingestor = BRIGHTIngestor(
            mock_embedding_model,
            domains=["biology"],
            document_mode="long",
        )
        assert ingestor.document_mode == "long"

    def test_init_invalid_domain_raises_error(self, mock_embedding_model):
        with pytest.raises(ValueError, match="Invalid domain 'invalid_domain'"):
            BRIGHTIngestor(mock_embedding_model, domains=["invalid_domain"])

    def test_init_long_mode_with_unsupported_domain_raises_error(self, mock_embedding_model):
        with pytest.raises(ValueError, match="does not support long documents"):
            BRIGHTIngestor(mock_embedding_model, domains=["leetcode"], document_mode="long")

    def test_init_long_mode_with_all_unsupported_domains_raises_error(self, mock_embedding_model):
        for domain in DOMAINS_WITHOUT_LONG_DOCS:
            with pytest.raises(ValueError, match="does not support long documents"):
                BRIGHTIngestor(mock_embedding_model, domains=[domain], document_mode="long")

    def test_init_long_mode_with_supported_domains_succeeds(self, mock_embedding_model):
        for domain in DOMAINS_WITH_LONG_DOCS:
            ingestor = BRIGHTIngestor(mock_embedding_model, domains=[domain], document_mode="long")
            assert ingestor.document_mode == "long"


class TestBRIGHTIngestorValidateDomains:
    def test_validate_all_bright_domains_valid(self, mock_embedding_model):
        ingestor = BRIGHTIngestor(mock_embedding_model)
        assert all(d in BRIGHT_DOMAINS for d in ingestor.domains)

    def test_validate_mixed_valid_invalid_domains_raises_error(self, mock_embedding_model):
        with pytest.raises(ValueError, match="Invalid domain"):
            BRIGHTIngestor(mock_embedding_model, domains=["biology", "not_a_domain"])


# ==================== Primary Key Type Tests ====================


class TestBRIGHTIngestorDetectPrimaryKeyType:
    def test_detect_primary_key_type_returns_string(self, mock_embedding_model):
        ingestor = BRIGHTIngestor(mock_embedding_model)
        assert ingestor.detect_primary_key_type() == "string"


# ==================== Helper Function Tests ====================


class TestMakeQueryId:
    def test_make_query_id_basic(self):
        result = _make_query_id("biology", "123")
        assert result == "biology_123"

    def test_make_query_id_with_complex_id(self):
        result = _make_query_id("stackoverflow", "question_456_abc")
        assert result == "stackoverflow_question_456_abc"


class TestMakeChunkId:
    def test_make_chunk_id_basic(self):
        result = _make_chunk_id("economics", "doc_789")
        assert result == "economics_doc_789"

    def test_make_chunk_id_with_numeric_source_id(self):
        result = _make_chunk_id("robotics", "12345")
        assert result == "robotics_12345"


class TestGetGoldIds:
    def test_get_gold_ids_short_mode(self):
        example = {"gold_ids": ["id1", "id2"], "gold_ids_long": ["long_id1"]}
        result = _get_gold_ids(example, "short", "biology")
        assert result == ["id1", "id2"]

    def test_get_gold_ids_long_mode(self):
        example = {"gold_ids": ["id1", "id2"], "gold_ids_long": ["long_id1", "long_id2"]}
        result = _get_gold_ids(example, "long", "biology")
        assert result == ["long_id1", "long_id2"]

    def test_get_gold_ids_long_mode_filters_na(self):
        example = {"gold_ids": ["id1"], "gold_ids_long": ["long_id1", "N/A", "long_id2"]}
        result = _get_gold_ids(example, "long", "biology")
        assert result == ["long_id1", "long_id2"]

    def test_get_gold_ids_long_mode_all_na_returns_empty(self):
        example = {"gold_ids": ["id1"], "gold_ids_long": ["N/A"]}
        result = _get_gold_ids(example, "long", "biology")
        assert result == []

    def test_get_gold_ids_long_mode_unsupported_domain_raises_error(self):
        example = {"gold_ids": ["id1"], "gold_ids_long": ["long_id1"]}
        with pytest.raises(ValueError, match="does not have long documents"):
            _get_gold_ids(example, "long", "leetcode")


# ==================== Ingest Flow Tests ====================


class TestBRIGHTIngestorIngest:
    @patch("autorag_research.data.bright.load_dataset")
    def test_ingest_raises_error_without_service(self, mock_load_dataset, mock_embedding_model):
        ingestor = BRIGHTIngestor(mock_embedding_model, domains=["biology"])
        with pytest.raises(ServiceNotSetError):
            ingestor.ingest()

    @patch("autorag_research.data.bright.load_dataset")
    def test_ingest_persists_data_to_database(
        self, mock_load_dataset, mock_embedding_model, text_service, sample_corpus_data, sample_examples_data
    ):
        # New flow: examples (queries) first, then documents (corpus)
        mock_load_dataset.side_effect = [
            create_mock_dataset(sample_examples_data),
            create_mock_dataset(sample_corpus_data),
        ]

        ingestor = BRIGHTIngestor(mock_embedding_model, domains=["biology"])
        ingestor.set_service(text_service)
        ingestor.ingest()

        stats = text_service.get_statistics()
        assert stats["chunks"]["total"] == 3
        assert stats["queries"]["total"] == 2

    @patch("autorag_research.data.bright.load_dataset")
    def test_ingest_processes_multiple_domains(
        self, mock_load_dataset, mock_embedding_model, text_service, sample_corpus_data, sample_examples_data
    ):
        # New flow: examples (queries) first, then documents (corpus) for each domain
        mock_load_dataset.side_effect = [
            create_mock_dataset(sample_examples_data),
            create_mock_dataset(sample_corpus_data),
            create_mock_dataset(sample_examples_data),
            create_mock_dataset(sample_corpus_data),
        ]

        ingestor = BRIGHTIngestor(mock_embedding_model, domains=["biology", "economics"])
        ingestor.set_service(text_service)
        ingestor.ingest()

        assert mock_load_dataset.call_count == 4
        stats = text_service.get_statistics()
        assert stats["chunks"]["total"] == 6
        assert stats["queries"]["total"] == 4


# ==================== Embed All Tests ====================


class TestBRIGHTIngestorEmbedAll:
    def test_embed_all_raises_error_without_service(self, mock_embedding_model):
        ingestor = BRIGHTIngestor(mock_embedding_model, domains=["biology"])
        with pytest.raises(ServiceNotSetError):
            ingestor.embed_all()

    @patch("autorag_research.data.bright.load_dataset")
    def test_embed_all_embeds_queries_and_chunks(
        self, mock_load_dataset, mock_embedding_model, text_service, sample_corpus_data, sample_examples_data
    ):
        # New flow: examples (queries) first, then documents (corpus)
        mock_load_dataset.side_effect = [
            create_mock_dataset(sample_examples_data),
            create_mock_dataset(sample_corpus_data),
        ]

        ingestor = BRIGHTIngestor(mock_embedding_model, domains=["biology"])
        ingestor.set_service(text_service)
        ingestor.ingest()
        ingestor.embed_all(max_concurrency=2, batch_size=10)

        stats = text_service.get_statistics()
        assert stats["chunks"]["with_embeddings"] == 3
        assert stats["chunks"]["without_embeddings"] == 0


# ==================== Set Service Tests ====================


class TestBRIGHTIngestorSetService:
    def test_set_service_stores_service(self, mock_embedding_model, text_service):
        ingestor = BRIGHTIngestor(mock_embedding_model, domains=["biology"])
        assert ingestor.service is None

        ingestor.set_service(text_service)
        assert ingestor.service is text_service


# ==================== Integration-Style Tests ====================


class TestBRIGHTIngestorIntegration:
    @patch("autorag_research.data.bright.load_dataset")
    def test_full_ingest_flow_short_mode(
        self,
        mock_load_dataset,
        mock_embedding_model,
        text_service,
        sample_corpus_data,
        sample_examples_data,
        bright_test_db,
        bright_db_session,
    ):
        # New flow: examples (queries) first, then documents (corpus)
        mock_load_dataset.side_effect = [
            create_mock_dataset(sample_examples_data),
            create_mock_dataset(sample_corpus_data),
        ]

        ingestor = BRIGHTIngestor(mock_embedding_model, domains=["biology"], document_mode="short")
        ingestor.set_service(text_service)
        ingestor.ingest()

        Chunk = bright_test_db["schema"].Chunk
        Query = bright_test_db["schema"].Query

        chunks = bright_db_session.query(Chunk).filter(Chunk.id.like("biology_%")).all()
        assert len(chunks) == 3
        assert all("biology_" in chunk.id for chunk in chunks)

        queries = bright_db_session.query(Query).filter(Query.id.like("biology_%")).all()
        assert len(queries) == 2
        assert queries[0].contents == "What is the first document about?"

    @patch("autorag_research.data.bright.load_dataset")
    def test_full_ingest_flow_long_mode(
        self, mock_load_dataset, mock_embedding_model, text_service, sample_long_corpus_data, sample_examples_data
    ):
        # New flow: examples (queries) first, then documents (corpus)
        mock_load_dataset.side_effect = [
            create_mock_dataset(sample_examples_data),
            create_mock_dataset(sample_long_corpus_data),
        ]

        ingestor = BRIGHTIngestor(mock_embedding_model, domains=["biology"], document_mode="long")
        ingestor.set_service(text_service)
        ingestor.ingest()

        mock_load_dataset.assert_any_call(
            "xlangai/BRIGHT", "long_documents", split="biology", streaming=True, trust_remote_code=True
        )

    @patch("autorag_research.data.bright.load_dataset")
    def test_ingest_with_batch_processing(self, mock_load_dataset, mock_embedding_model, text_service):
        large_corpus = [{"id": f"doc_{i:04d}", "content": f"Content {i}"} for i in range(2500)]
        large_examples = [
            {
                "id": f"q_{i:04d}",
                "query": f"Query {i}",
                "gold_ids": [f"doc_{i:04d}"],
                "gold_ids_long": [f"long_doc_{i:04d}"],
                "gold_answer": f"Answer {i}",
                "reasoning": "Some reasoning",
                "excluded_ids": [],
            }
            for i in range(100)
        ]

        # New flow: examples (queries) first, then documents (corpus)
        mock_load_dataset.side_effect = [
            create_mock_dataset(large_examples),
            create_mock_dataset(large_corpus),
        ]

        ingestor = BRIGHTIngestor(mock_embedding_model, domains=["biology"])
        ingestor.set_service(text_service)
        ingestor.ingest()

        stats = text_service.get_statistics()
        assert stats["chunks"]["total"] == 2500
        assert stats["queries"]["total"] == 100
