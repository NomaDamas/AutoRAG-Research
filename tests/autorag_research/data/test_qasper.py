"""Tests for QasperIngestor.

Integration tests use real Qasper dataset subsets against PostgreSQL.
"""

import pytest
from llama_index.core import MockEmbedding

from autorag_research.data.qasper import (
    QasperIngestor,
    _find_chunk_ids_for_evidence,
    _iter_paper_chunks,
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


class TestIterPaperChunks:
    def test_iter_handles_multiple_sections_and_empty_paragraphs(self):
        """Test iteration with multiple sections, empty paragraphs, and whitespace."""
        paragraphs = [
            ["Intro.", "", None],
            ["  Methods 1.  ", "Methods 2."]
        ]
        result = list(_iter_paper_chunks("paper123", paragraphs))

        assert len(result) == 3
        assert result[0] == ("qasper_paper123_s0_p0", "Intro.")
        assert result[1] == ("qasper_paper123_s1_p0", "Methods 1.")
        assert result[2] == ("qasper_paper123_s1_p1", "Methods 2.")


class TestFindChunkIdsForEvidence:
    def test_find_partial_match(self):
        """Test partial matching when evidence is substring of chunk."""
        evidence_map = {
            "this is a longer paragraph with more text.": "chunk_1",
        }
        evidence_list = ["longer paragraph"]
        result = _find_chunk_ids_for_evidence(evidence_list, evidence_map)
        assert result == ["chunk_1"]


# ==================== Integration Tests ====================


QASPER_ANSWERABLE_CONFIG = IngestorTestConfig(
    expected_query_count=10,
    expected_chunk_count=50,
    chunk_count_is_minimum=True,
    check_retrieval_relations=True,
    check_generation_gt=True,
    generation_gt_required_for_all=True,
    primary_key_type="string",
    db_name="qasper_answerable_test",
)


QASPER_FULL_CONFIG = IngestorTestConfig(
    expected_query_count=15,
    expected_chunk_count=50,
    chunk_count_is_minimum=True,
    check_retrieval_relations=False,  # full mode does not have retrieval GT
    check_generation_gt=True,
    generation_gt_required_for_all=True,
    primary_key_type="string",
    db_name="qasper_full_test",
)


@pytest.mark.data
class TestQasperIngestorIntegration:
    """Integration tests using real Qasper dataset subsets."""

    def test_ingest_answerable_mode(self, mock_embedding_model):
        """Test ingestion with answerable mode (Retrieval + Generation evaluation)."""
        with create_test_database(QASPER_ANSWERABLE_CONFIG) as db:
            service = TextDataIngestionService(db.session_factory, schema=db.schema)

            ingestor = QasperIngestor(mock_embedding_model, qa_mode="answerable")
            ingestor.set_service(service)
            ingestor.ingest(
                query_limit=QASPER_ANSWERABLE_CONFIG.expected_query_count,
                min_corpus_cnt=QASPER_ANSWERABLE_CONFIG.expected_chunk_count,
            )

            verifier = IngestorTestVerifier(service, db.schema, QASPER_ANSWERABLE_CONFIG)
            verifier.verify_all()

    def test_ingest_full_mode(self, mock_embedding_model):
        """Test ingestion with full mode (Generation evaluation only)."""
        with create_test_database(QASPER_FULL_CONFIG) as db:
            service = TextDataIngestionService(db.session_factory, schema=db.schema)

            ingestor = QasperIngestor(mock_embedding_model, qa_mode="full")
            ingestor.set_service(service)
            ingestor.ingest(
                query_limit=QASPER_FULL_CONFIG.expected_query_count,
                min_corpus_cnt=QASPER_FULL_CONFIG.expected_chunk_count,
            )

            verifier = IngestorTestVerifier(service, db.schema, QASPER_FULL_CONFIG)
            verifier.verify_all()

    def test_embed_all(self, mock_embedding_model):
        """Test embedding after ingestion."""
        config = IngestorTestConfig(
            expected_query_count=5,
            expected_chunk_count=25,
            chunk_count_is_minimum=True,
            check_retrieval_relations=True,
            check_generation_gt=True,
            generation_gt_required_for_all=True,
            primary_key_type="string",
            db_name="qasper_embed_test",
        )
        with create_test_database(config) as db:
            service = TextDataIngestionService(db.session_factory, schema=db.schema)

            ingestor = QasperIngestor(mock_embedding_model, qa_mode="answerable")
            ingestor.set_service(service)
            ingestor.ingest(
                query_limit=config.expected_query_count,
                min_corpus_cnt=config.expected_chunk_count,
            )

            # Verify embedding works
            ingestor.embed_all(max_concurrency=4, batch_size=10)

            stats = service.get_statistics()
            assert stats["chunks"]["with_embeddings"] == stats["chunks"]["total"]
