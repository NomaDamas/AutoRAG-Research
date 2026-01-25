"""Tests for BRIGHTIngestor.

Unit tests for helper functions and constructor validation.
Integration tests use real data subsets against PostgreSQL.
"""

import pytest
from llama_index.core import MockEmbedding

from autorag_research.data.bright import (
    BRIGHTIngestor,
    _get_gold_ids,
    _make_id,
)
from autorag_research.orm.service.text_ingestion import TextDataIngestionService
from tests.autorag_research.data.ingestor_test_utils import (
    IngestorTestConfig,
    IngestorTestVerifier,
    create_test_database,
)

EMBEDDING_DIM = 768


# ==================== Fixtures ====================


@pytest.fixture
def mock_embedding_model():
    return MockEmbedding(EMBEDDING_DIM)


# ==================== Unit Tests: Helper Functions ====================
class TestMakeId:
    def test_make_id_basic(self):
        result = _make_id("economics", "doc_789")
        assert result == "economics_doc_789"

    def test_make_id_with_numeric_source_id(self):
        result = _make_id("robotics", "12345")
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


# ==================== Integration Tests ====================


BRIGHT_INTEGRATION_CONFIG = IngestorTestConfig(
    expected_query_count=10,
    expected_chunk_count=50,
    chunk_count_is_minimum=True,  # Gold IDs always included, actual count may exceed limit
    check_retrieval_relations=True,
    check_generation_gt=True,
    primary_key_type="string",
    db_name="bright_integration_test",
)


@pytest.mark.data
class TestBRIGHTIngestorIntegration:
    """Integration tests using real BRIGHT dataset subsets."""

    def test_ingest_biology_domain(self, mock_embedding_model):
        """Test ingestion of biology domain with query and corpus limits."""
        with create_test_database(BRIGHT_INTEGRATION_CONFIG) as db:
            service = TextDataIngestionService(db.session_factory, schema=db.schema)

            ingestor = BRIGHTIngestor(
                mock_embedding_model,
                domain="biology",
                document_mode="short",
            )
            ingestor.set_service(service)
            ingestor.ingest(
                query_limit=BRIGHT_INTEGRATION_CONFIG.expected_query_count,
                min_corpus_cnt=BRIGHT_INTEGRATION_CONFIG.expected_chunk_count,
            )

            verifier = IngestorTestVerifier(service, db.schema, BRIGHT_INTEGRATION_CONFIG)
            verifier.verify_all()

    def test_ingest_long_documents(self, mock_embedding_model):
        """Test ingestion with long document mode."""
        config = IngestorTestConfig(
            expected_query_count=5,
            expected_chunk_count=25,
            chunk_count_is_minimum=True,
            check_retrieval_relations=True,
            check_generation_gt=True,
            primary_key_type="string",
            db_name="bright_long_docs_test",
        )

        with create_test_database(config) as db:
            service = TextDataIngestionService(db.session_factory, schema=db.schema)

            ingestor = BRIGHTIngestor(
                mock_embedding_model,
                domain="biology",
                document_mode="long",
            )
            ingestor.set_service(service)
            ingestor.ingest(
                query_limit=config.expected_query_count,
                min_corpus_cnt=config.expected_chunk_count,
            )

            verifier = IngestorTestVerifier(service, db.schema, config)
            verifier.verify_all()
