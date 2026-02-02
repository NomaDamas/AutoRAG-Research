"""Tests for TextMTEBDatasetIngestor.

Unit tests for helper functions and task validation.
Integration tests use real MTEB dataset subsets against PostgreSQL.
"""

import pandas as pd
import pytest
from langchain_core.embeddings.fake import FakeEmbeddings

from autorag_research.data.text_mteb import (
    SUPPORTED_TASK_TYPES,
    TextMTEBDatasetIngestor,
    _combine_title_text,
)
from autorag_research.exceptions import UnsupportedMTEBTaskTypeError
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


# ==================== Unit Tests: Helper Functions ====================


class TestCombineTitleText:
    def test_combine_with_both(self):
        row = pd.Series({"title": "The Title", "text": "The content"})
        result = _combine_title_text(row)
        assert result == "The Title\nThe content"

    def test_combine_text_only(self):
        row = pd.Series({"text": "The content"})
        result = _combine_title_text(row)
        assert result == "The content"

    def test_combine_empty_title(self):
        row = pd.Series({"title": "", "text": "The content"})
        result = _combine_title_text(row)
        assert result == "The content"

    def test_combine_whitespace_title(self):
        row = pd.Series({"title": "   ", "text": "The content"})
        result = _combine_title_text(row)
        assert result == "The content"

    def test_combine_with_none_title(self):
        row = pd.Series({"title": None, "text": "The content"})
        result = _combine_title_text(row)
        assert result == "The content"


# ==================== Unit Tests: Constructor ====================


class TestTextMTEBDatasetIngestorInit:
    @pytest.mark.ci_skip
    def test_detect_primary_key_type_nfcorpus(self, mock_embedding_model):
        """NFCorpus uses string primary keys."""
        ingestor = TextMTEBDatasetIngestor(mock_embedding_model, "NFCorpus")
        assert ingestor.detect_primary_key_type() == "string"

    def test_task_name_stored_directly(self, mock_embedding_model):
        """Task name should be stored as provided without normalization."""
        ingestor = TextMTEBDatasetIngestor(mock_embedding_model, "NFCorpus")
        assert ingestor.task_name == "NFCorpus"

    def test_include_instruction_default_true(self, mock_embedding_model):
        """include_instruction should default to True."""
        ingestor = TextMTEBDatasetIngestor(mock_embedding_model, "NFCorpus")
        assert ingestor.include_instruction is True

    def test_include_instruction_can_be_disabled(self, mock_embedding_model):
        """include_instruction can be set to False."""
        ingestor = TextMTEBDatasetIngestor(mock_embedding_model, "NFCorpus", include_instruction=False)
        assert ingestor.include_instruction is False


# ==================== Unit Tests: Task Validation ====================


class TestTextMTEBDatasetIngestorValidation:
    @pytest.mark.ci_skip
    def test_retrieval_task_is_supported(self, mock_embedding_model):
        """Standard Retrieval task should be supported."""
        ingestor = TextMTEBDatasetIngestor(mock_embedding_model, "NFCorpus")
        # Should not raise
        ingestor._validate_task()

    def test_supported_task_types_contains_expected(self):
        """Verify SUPPORTED_TASK_TYPES contains all expected types."""
        expected = {"Retrieval", "InstructionRetrieval"}
        assert expected == SUPPORTED_TASK_TYPES

    @pytest.mark.data
    def test_unsupported_task_type_raises_error(self, mock_embedding_model):
        """Clustering task should raise error."""
        # ArxivClusteringP2P is a Clustering task
        ingestor = TextMTEBDatasetIngestor(mock_embedding_model, "ArxivClusteringP2P")
        with pytest.raises(UnsupportedMTEBTaskTypeError) as exc_info:
            ingestor._validate_task()
        assert "Clustering" in str(exc_info.value)
        assert "not supported" in str(exc_info.value)


# ==================== Integration Tests ====================


MTEB_SMALL_CONFIG = IngestorTestConfig(
    expected_query_count=10,
    expected_chunk_count=50,
    chunk_count_is_minimum=True,  # Gold IDs always included, actual count may exceed limit
    check_retrieval_relations=True,
    check_generation_gt=False,  # MTEB retrieval datasets don't have generation ground truth
    primary_key_type="string",
    db_name="mteb_nfcorpus_test",
)


@pytest.mark.data
class TestTextMTEBDatasetIngestorIntegration:
    """Integration tests using real MTEB dataset subsets."""

    def test_ingest_nfcorpus_subset(self, mock_embedding_model):
        """Test ingestion of NFCorpus dataset with query and corpus limits."""
        with create_test_database(MTEB_SMALL_CONFIG) as db:
            service = TextDataIngestionService(db.session_factory, schema=db.schema)

            ingestor = TextMTEBDatasetIngestor(mock_embedding_model, "NFCorpus")
            ingestor.set_service(service)
            ingestor.ingest(
                subset="test",
                query_limit=MTEB_SMALL_CONFIG.expected_query_count,
                min_corpus_cnt=MTEB_SMALL_CONFIG.expected_chunk_count,
            )

            verifier = IngestorTestVerifier(service, db.schema, MTEB_SMALL_CONFIG)
            verifier.verify_all()

    def test_if_ir_nfcorpus_subset(self, mock_embedding_model):
        config = IngestorTestConfig(
            expected_query_count=10,
            expected_chunk_count=50,
            chunk_count_is_minimum=True,
            check_retrieval_relations=True,
            check_generation_gt=False,
            primary_key_type="string",
            db_name="mteb_if_ir_nfcorpus_test",
        )
        with create_test_database(config) as db:
            service = TextDataIngestionService(db.session_factory, schema=db.schema)

            ingestor = TextMTEBDatasetIngestor(mock_embedding_model, "IFIRNFCorpus")
            ingestor.set_service(service)
            ingestor.ingest(
                subset="test",
                query_limit=config.expected_query_count,
                min_corpus_cnt=config.expected_chunk_count,
            )

            verifier = IngestorTestVerifier(service, db.schema, config)
            verifier.verify_all()

    def test_ingest_scifact_subset(self, mock_embedding_model):
        """Test ingestion of SciFact dataset with query and corpus limits."""
        config = IngestorTestConfig(
            expected_query_count=10,
            expected_chunk_count=50,
            chunk_count_is_minimum=True,
            check_retrieval_relations=True,
            check_generation_gt=False,
            primary_key_type="string",
            db_name="mteb_scifact_test",
        )

        with create_test_database(config) as db:
            service = TextDataIngestionService(db.session_factory, schema=db.schema)

            ingestor = TextMTEBDatasetIngestor(mock_embedding_model, "SciFact")
            ingestor.set_service(service)
            ingestor.ingest(
                subset="test",
                query_limit=config.expected_query_count,
                min_corpus_cnt=config.expected_chunk_count,
            )

            verifier = IngestorTestVerifier(service, db.schema, config)
            verifier.verify_all()

    def test_ingest_with_score_threshold(self, mock_embedding_model):
        """Test ingestion with custom score threshold."""
        config = IngestorTestConfig(
            expected_query_count=5,
            expected_chunk_count=25,
            chunk_count_is_minimum=True,
            check_retrieval_relations=True,
            check_generation_gt=False,
            primary_key_type="string",
            db_name="mteb_threshold_test",
        )

        with create_test_database(config) as db:
            service = TextDataIngestionService(db.session_factory, schema=db.schema)

            ingestor = TextMTEBDatasetIngestor(mock_embedding_model, "NFCorpus", score_threshold=2)
            ingestor.set_service(service)
            ingestor.ingest(
                subset="test",
                query_limit=config.expected_query_count,
                min_corpus_cnt=config.expected_chunk_count,
            )

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
            db_name="mteb_embed_test",
        )

        with create_test_database(config) as db:
            service = TextDataIngestionService(db.session_factory, schema=db.schema)

            ingestor = TextMTEBDatasetIngestor(mock_embedding_model, "NFCorpus")
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
