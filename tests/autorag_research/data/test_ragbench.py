"""Tests for RAGBenchIngestor.

Unit tests for helper functions (TDD - tests written before implementation).
Integration tests use real data subsets against PostgreSQL.
"""

import pytest
from llama_index.core import MockEmbedding

from autorag_research.data.ragbench import (
    RAGBenchIngestor,
    _make_query_id,
    compute_chunk_id,
    extract_relevant_doc_indices,
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


class TestExtractRelevantDocIndices:
    def test_extract_single_digit_indices(self):
        sentence_keys = ["0a", "0b", "1a", "2c"]
        result = extract_relevant_doc_indices(sentence_keys)
        assert result == {0, 1, 2}

    def test_extract_multi_digit_indices(self):
        sentence_keys = ["10a", "12b", "0c", "5d"]
        result = extract_relevant_doc_indices(sentence_keys)
        assert result == {0, 5, 10, 12}

    def test_extract_duplicate_doc_indices(self):
        sentence_keys = ["0a", "0b", "0c", "0d", "0e", "0f"]
        result = extract_relevant_doc_indices(sentence_keys)
        assert result == {0}

    def test_extract_empty_list(self):
        result = extract_relevant_doc_indices([])
        assert result == set()

    def test_extract_mixed_indices(self):
        sentence_keys = ["0d", "0e", "0f", "1d", "1e", "1f"]
        result = extract_relevant_doc_indices(sentence_keys)
        assert result == {0, 1}


class TestComputeChunkId:
    def test_compute_chunk_id_basic(self):
        content = "Title: Test Document\nPassage: This is a test passage."
        config = "covidqa"
        result = compute_chunk_id(content, config)
        assert result.startswith("covidqa_")
        assert len(result) == len("covidqa_") + 16  # config + underscore + 16 char hash

    def test_compute_chunk_id_same_content_same_hash(self):
        content = "Title: Same\nPassage: Same content"
        config = "covidqa"
        result1 = compute_chunk_id(content, config)
        result2 = compute_chunk_id(content, config)
        assert result1 == result2

    def test_compute_chunk_id_different_content_different_hash(self):
        config = "covidqa"
        result1 = compute_chunk_id("Content A", config)
        result2 = compute_chunk_id("Content B", config)
        assert result1 != result2

    def test_compute_chunk_id_different_config_different_hash(self):
        content = "Same content"
        result1 = compute_chunk_id(content, "covidqa")
        result2 = compute_chunk_id(content, "hotpotqa")
        assert result1 != result2
        assert result1.startswith("covidqa_")
        assert result2.startswith("hotpotqa_")

    def test_compute_chunk_id_normalization(self):
        config = "covidqa"
        result1 = compute_chunk_id("  Content   with   spaces  ", config)
        result2 = compute_chunk_id("content with spaces", config)
        assert result1 == result2

    def test_compute_chunk_id_case_insensitive(self):
        config = "covidqa"
        result1 = compute_chunk_id("CONTENT", config)
        result2 = compute_chunk_id("content", config)
        assert result1 == result2


class TestMakeQueryId:
    def test_make_query_id_basic(self):
        result = _make_query_id("covidqa", "train", "358")
        assert result == "covidqa_train_358"

    def test_make_query_id_different_split(self):
        result = _make_query_id("hotpotqa", "test", "123")
        assert result == "hotpotqa_test_123"

    def test_make_query_id_validation_split(self):
        result = _make_query_id("finqa", "validation", "456")
        assert result == "finqa_validation_456"


# ==================== Integration Tests ====================


RAGBENCH_INTEGRATION_CONFIG = IngestorTestConfig(
    expected_query_count=10,
    expected_chunk_count=30,  # ~3-4 docs per query, some deduplication
    chunk_count_is_minimum=True,  # Exact count depends on deduplication
    check_retrieval_relations=True,
    check_generation_gt=True,
    generation_gt_required_for_all=True,  # RAGbench has responses for all queries
    primary_key_type="string",
    db_name="ragbench_integration_test",
)


@pytest.mark.data
class TestRAGBenchIngestorIntegration:
    def test_ingest_covidqa_subset(self, mock_embedding_model):
        with create_test_database(RAGBENCH_INTEGRATION_CONFIG) as db:
            service = TextDataIngestionService(db.session_factory, schema=db.schema)

            ingestor = RAGBenchIngestor(
                mock_embedding_model,
                configs=["covidqa"],
            )
            ingestor.set_service(service)
            ingestor.ingest(
                query_limit=RAGBENCH_INTEGRATION_CONFIG.expected_query_count,
                subset="test",
            )

            verifier = IngestorTestVerifier(service, db.schema, RAGBENCH_INTEGRATION_CONFIG)
            verifier.verify_all()
