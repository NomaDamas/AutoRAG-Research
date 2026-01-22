"""Tests for DocVQA Ingestor.

Integration tests use real data subsets against PostgreSQL.
DocVQA is a multi-modal dataset with document images and text queries.
"""

import pytest

from autorag_research.data.docvqa import DocVQAIngestor
from autorag_research.orm.service.multi_modal_ingestion import MultiModalIngestionService
from tests.autorag_research.data.ingestor_test_utils import (
    IngestorTestConfig,
    IngestorTestVerifier,
    create_test_database,
)

# ==================== Integration Tests ====================

DOCVQA_INTEGRATION_CONFIG = IngestorTestConfig(
    expected_query_count=10,
    expected_image_chunk_count=10,  # May be higher if corpus_limit > query_limit
    chunk_count_is_minimum=True,  # Gold IDs always included
    check_retrieval_relations=True,
    check_generation_gt=True,
    generation_gt_required_for_all=True,  # All queries have answers
    primary_key_type="string",
    db_name="docvqa_test",
)


@pytest.mark.data
class TestDocVQAIngestorIntegration:
    def test_ingest_subset(self):
        with create_test_database(DOCVQA_INTEGRATION_CONFIG) as db:
            service = MultiModalIngestionService(db.session_factory, schema=db.schema)

            ingestor = DocVQAIngestor()
            ingestor.set_service(service)
            ingestor.ingest(
                query_limit=DOCVQA_INTEGRATION_CONFIG.expected_query_count,
                corpus_limit=DOCVQA_INTEGRATION_CONFIG.expected_image_chunk_count,
            )

            verifier = IngestorTestVerifier(service, db.schema, DOCVQA_INTEGRATION_CONFIG)
            verifier.verify_all()
