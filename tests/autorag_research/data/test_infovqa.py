"""Tests for InfoVQA Ingestor.

Integration tests for the InfoVQA (openbmb/VisRAG-Ret-Test-InfoVQA) dataset ingestor.
Uses real data subsets against PostgreSQL.

Dataset characteristics:
- Multi-modal: text queries retrieve infographic images
- String primary keys (corpus-id like "36966.jpeg", query-id like "36966.jpeg-1")
- 1:1 query-to-image mapping via explicit qrels
- All queries have generation_gt (answer lists)
"""

import pytest
from autorag_research.data.infovqa import InfoVQAIngestor

from autorag_research.orm.service.multi_modal_ingestion import MultiModalIngestionService
from tests.autorag_research.data.ingestor_test_utils import (
    IngestorTestConfig,
    IngestorTestVerifier,
    create_test_database,
)

# ==================== Integration Tests ====================

CONFIG = IngestorTestConfig(
    expected_query_count=10,
    expected_image_chunk_count=10,  # 1:1 mapping with queries
    check_retrieval_relations=True,
    check_generation_gt=True,
    generation_gt_required_for_all=True,  # All queries have answers
    primary_key_type="string",
    db_name="infovqa_test",
)


@pytest.mark.data
class TestInfoVQAIngestorIntegration:
    """Integration tests using real InfoVQA dataset subsets."""

    def test_ingest_subset(self):
        """Basic integration test - verify_all() handles all standard checks.

        Verifies:
        - Query count matches expected (10)
        - Image chunk count matches expected (10, 1:1 with queries)
        - All queries have retrieval relations
        - All queries have generation_gt
        - Query/image format validation (IDs, content types, mimetype)
        """
        with create_test_database(CONFIG) as db:
            service = MultiModalIngestionService(db.session_factory, schema=db.schema)

            ingestor = InfoVQAIngestor()
            ingestor.set_service(service)
            ingestor.ingest(
                query_limit=CONFIG.expected_query_count,
                corpus_limit=CONFIG.expected_image_chunk_count,
            )

            verifier = IngestorTestVerifier(service, db.schema, CONFIG)
            verifier.verify_all()
