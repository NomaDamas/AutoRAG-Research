"""Tests for ChartQA Ingestor.

Integration tests using VisRAG filtered ChartQA dataset against PostgreSQL.

VisRAG-Ret-Test-ChartQA filters context-dependent questions using GPT-4o judge,
preserving ~10% of the original dataset for better retrieval evaluation quality.

Key characteristics:
- Uses string primary keys (corpus-id format like 'chartqa/test/png/...')
- Separate corpus (500 images), queries (63), and qrels
- Each query maps to one or more corpus images via qrels
- Answer field provides generation ground truth
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
    expected_image_chunk_count=10,  # At minimum, gold images for queries
    chunk_count_is_minimum=True,  # May have more images due to gold ID preservation
    check_retrieval_relations=True,
    check_generation_gt=True,
    generation_gt_required_for_all=True,  # All queries have answers
    primary_key_type="string",  # VisRAG uses string IDs
    db_name="chartqa_integration_test",
)


@pytest.mark.data
class TestChartQAIngestorIntegration:
    """Integration tests using VisRAG filtered ChartQA dataset."""

    def test_ingest_subset(self):
        """Basic integration test - verify_all() handles all standard checks.

        This test verifies:
        - Correct query count (10)
        - Image chunk count (at least 10, gold images always included)
        - All queries have retrieval relations
        - All queries have generation_gt
        - Query format validation (non-empty strings)
        - Image chunk format validation (valid image bytes, mimetype)
        """
        with create_test_database(CHARTQA_INTEGRATION_CONFIG) as db:
            service = MultiModalIngestionService(db.session_factory, schema=db.schema)

            ingestor = ChartQAIngestor()
            ingestor.set_service(service)
            ingestor.ingest(
                query_limit=CHARTQA_INTEGRATION_CONFIG.expected_query_count,
                corpus_limit=CHARTQA_INTEGRATION_CONFIG.expected_image_chunk_count,
            )

            verifier = IngestorTestVerifier(service, db.schema, CHARTQA_INTEGRATION_CONFIG)
            verifier.verify_all()
