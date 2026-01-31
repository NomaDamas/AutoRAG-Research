"""Tests for BEIRIngestor.

Integration tests use real BEIR dataset subsets against PostgreSQL.
"""

import pytest
from langchain_core.embeddings.fake import FakeEmbeddings

from autorag_research.data.beir import BEIRIngestor
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
    return FakeEmbeddings(size=EMBEDDING_DIM)


# ==================== Unit Tests ====================


class TestBEIRIngestorInit:
    def test_detect_primary_key_type_scifact(self, mock_embedding_model):
        """SciFact dataset uses string primary keys."""
        ingestor = BEIRIngestor(mock_embedding_model, "scifact")
        assert ingestor.detect_primary_key_type() == "string"


# ==================== Integration Tests ====================


BEIR_SMALL_CONFIG = IngestorTestConfig(
    expected_query_count=10,
    expected_chunk_count=50,
    chunk_count_is_minimum=True,  # Gold IDs always included, actual count may exceed limit
    check_retrieval_relations=True,
    check_generation_gt=False,  # BEIR datasets don't have generation ground truth
    primary_key_type="string",
    db_name="beir_scifact_test",
)


@pytest.mark.data
class TestBEIRIngestorIntegration:
    """Integration tests using real BEIR dataset subsets."""

    def test_ingest_scifact_subset(self, mock_embedding_model):
        """Test ingestion of SciFact dataset with query and corpus limits."""
        with create_test_database(BEIR_SMALL_CONFIG) as db:
            service = TextDataIngestionService(db.session_factory, schema=db.schema)

            ingestor = BEIRIngestor(mock_embedding_model, "scifact")
            ingestor.set_service(service)
            ingestor.ingest(
                subset="test",
                query_limit=BEIR_SMALL_CONFIG.expected_query_count,
                min_corpus_cnt=BEIR_SMALL_CONFIG.expected_chunk_count,
            )

            verifier = IngestorTestVerifier(service, db.schema, BEIR_SMALL_CONFIG)
            verifier.verify_all()

    def test_ingest_scifact_full(self, mock_embedding_model):
        """Test full ingestion of SciFact dataset (all queries and corpus)."""
        config = IngestorTestConfig(
            expected_query_count=300,
            expected_chunk_count=5183,
            check_retrieval_relations=True,
            check_generation_gt=False,
            primary_key_type="string",
            db_name="beir_scifact_full_test",
        )

        with create_test_database(config) as db:
            service = TextDataIngestionService(db.session_factory, schema=db.schema)

            ingestor = BEIRIngestor(mock_embedding_model, "scifact")
            ingestor.set_service(service)
            ingestor.ingest(subset="test")

            verifier = IngestorTestVerifier(service, db.schema, config)
            verifier.verify_all()

    def test_embed_all(self, mock_embedding_model):
        """Test embedding after ingestion."""
        config = IngestorTestConfig(
            expected_query_count=5,
            expected_chunk_count=25,
            chunk_count_is_minimum=True,
            check_retrieval_relations=True,
            check_generation_gt=False,
            primary_key_type="string",
            db_name="beir_embed_test",
        )

        with create_test_database(config) as db:
            service = TextDataIngestionService(db.session_factory, schema=db.schema)

            ingestor = BEIRIngestor(mock_embedding_model, "scifact")
            ingestor.set_service(service)
            ingestor.ingest(
                subset="test",
                query_limit=config.expected_query_count,
                min_corpus_cnt=config.expected_chunk_count,
            )

            # Before embedding
            stats = service.get_statistics()
            assert stats["chunks"]["with_embeddings"] == 0

            # Run embedding
            ingestor.embed_all(max_concurrency=4, batch_size=10)

            # After embedding
            stats = service.get_statistics()
            assert stats["chunks"]["with_embeddings"] == stats["chunks"]["total"]
            assert stats["chunks"]["without_embeddings"] == 0
