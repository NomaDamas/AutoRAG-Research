"""Tests for OpenRAGBenchIngestor.

Unit tests for helper functions.
Integration tests use real data subsets against PostgreSQL.
"""

import pytest

from autorag_research.data.open_ragbench import OpenRAGBenchIngestor
from autorag_research.orm.service.multi_modal_ingestion import MultiModalIngestionService
from tests.autorag_research.data.ingestor_test_utils import (
    IngestorTestConfig,
    IngestorTestVerifier,
    create_test_database,
)

# ==================== Integration Tests ====================


OPENRAGBENCH_CONFIG = IngestorTestConfig(
    expected_query_count=10,
    expected_chunk_count=10,  # Each query maps to exactly one section
    expected_image_chunk_count=5,  # Some sections have images (figures/tables)
    chunk_count_is_minimum=True,  # Gold sections always included, may have more
    # Full hierarchy checks:
    #   - File: stores arxiv pdf_url
    #   - Chunk: Document -> Page -> Caption -> Chunk
    #   - ImageChunk: Document -> Page -> ImageChunk
    check_files=True,
    expected_file_count=10,  # One file per document (arxiv pdf_url)
    check_documents=True,
    expected_document_count=10,  # One document per query's gold section (unique doc_ids)
    check_pages=True,
    expected_page_count=10,  # One page per section with content (text or images)
    check_captions=True,
    expected_caption_count=10,  # One caption per section with text
    check_retrieval_relations=True,
    check_generation_gt=True,
    generation_gt_required_for_all=True,  # All queries have answers in answers.json
    primary_key_type="string",
    db_name="openragbench_integration_test",
)


@pytest.mark.data
class TestOpenRAGBenchIngestorIntegration:
    def test_ingest_subset(self):
        with create_test_database(OPENRAGBENCH_CONFIG) as db:
            service = MultiModalIngestionService(db.session_factory, schema=db.schema)

            ingestor = OpenRAGBenchIngestor()
            ingestor.set_service(service)
            ingestor.ingest(
                query_limit=OPENRAGBENCH_CONFIG.expected_query_count,
            )

            verifier = IngestorTestVerifier(service, db.schema, OPENRAGBENCH_CONFIG)
            verifier.verify_all()
