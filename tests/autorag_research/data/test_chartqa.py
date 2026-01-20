"""Tests for ChartQA Ingestor.

Integration tests using real ChartQA dataset subsets against PostgreSQL.

ChartQA is a visual question answering dataset with chart images.
Key characteristics:
- 1:1 query to image mapping
- Label field provides generation ground truth (list[str])
- Uses bigint primary keys (auto-generated)
- No special query formatting (raw questions)
"""

import pytest

from autorag_research.data.chartqa import ChartQAIngestor
from autorag_research.orm.service.multi_modal_ingestion import MultiModalIngestionService
from tests.autorag_research.data.ingestor_test_utils import (
    IngestorTestConfig,
    IngestorTestVerifier,
    create_test_database,
)

# ==================== Integration Tests ====================

CHARTQA_INTEGRATION_CONFIG = IngestorTestConfig(
    expected_query_count=10,
    expected_image_chunk_count=10,  # 1:1 query to image mapping
    chunk_count_is_minimum=False,  # Exact counts
    check_retrieval_relations=True,
    check_generation_gt=True,
    generation_gt_required_for_all=True,  # All queries have answers
    primary_key_type="bigint",
    db_name="chartqa_integration_test",
)


@pytest.mark.data
class TestChartQAIngestorIntegration:
    """Integration tests using real ChartQA dataset subsets."""

    def test_ingest_subset(self):
        """Basic integration test - verify_all() handles all standard checks.

        This test verifies:
        - Correct query count (10)
        - Correct image chunk count (10, 1:1 mapping)
        - All queries have retrieval relations
        - All queries have generation_gt
        - Query format validation (non-empty strings)
        - Image chunk format validation (valid image bytes, mimetype)
        """
        with create_test_database(CHARTQA_INTEGRATION_CONFIG) as db:
            service = MultiModalIngestionService(db.session_factory, schema=db.schema)

            ingestor = ChartQAIngestor()
            ingestor.set_service(service)
            ingestor.ingest(query_limit=CHARTQA_INTEGRATION_CONFIG.expected_query_count)

            verifier = IngestorTestVerifier(service, db.schema, CHARTQA_INTEGRATION_CONFIG)
            verifier.verify_all()
