"""Tests for ViDoReV2 Ingestors.

Unit tests for helper functions and constructor validation.
Integration tests use real data subsets against PostgreSQL.

ViDoReV2 uses BEIR-like structure with separate corpus, queries, qrels subsets.
Unlike V1, it supports many-to-many query-to-corpus relations.
"""

import pytest

from autorag_research.data.vidorev2 import (
    ViDoReV2DatasetName,
    ViDoReV2Ingestor,
)
from autorag_research.orm.service.multi_modal_ingestion import MultiModalIngestionService
from tests.autorag_research.data.ingestor_test_utils import (
    IngestorTestConfig,
    IngestorTestVerifier,
    create_test_database,
)

# ==================== Unit Tests: Dataset Validation ====================


class TestViDoReV2DatasetValidation:
    """Test dataset name validation for ViDoReV2Ingestor."""

    def test_valid_dataset_names(self):
        """Verify all valid dataset names are in the enum."""
        expected_datasets = [
            "esg_reports_v2",
            "biomedical_lectures_v2",
            "economics_reports_v2",
        ]
        actual_datasets = [d.value for d in ViDoReV2DatasetName]
        assert sorted(actual_datasets) == sorted(expected_datasets)


# ==================== Integration Tests ====================
VIDOREV2_ECONOMICS_REPORTS_CONFIG = IngestorTestConfig(
    expected_query_count=10,
    expected_image_chunk_count=50,  # Separate corpus from queries
    chunk_count_is_minimum=True,  # Gold IDs always included
    check_retrieval_relations=True,
    check_generation_gt=True,
    primary_key_type="string",
    db_name="vidorev2_economics_reports_test",
)


@pytest.mark.data
class TestViDoReV2ESGReportsIngestorIntegration:
    """Integration tests using real ViDoReV2 ESG reports dataset."""

    def test_ingest_economics_reports_v2_subset(self):
        """Test ingestion of esg_reports_v2 dataset with streaming."""
        with create_test_database(VIDOREV2_ECONOMICS_REPORTS_CONFIG) as db:
            service = MultiModalIngestionService(db.session_factory, schema=db.schema)

            ingestor = ViDoReV2Ingestor(ViDoReV2DatasetName.ESG_REPORTS_V2)
            ingestor.set_service(service)
            ingestor.ingest(
                query_limit=VIDOREV2_ECONOMICS_REPORTS_CONFIG.expected_query_count,
                corpus_limit=VIDOREV2_ECONOMICS_REPORTS_CONFIG.expected_image_chunk_count,
            )

            verifier = IngestorTestVerifier(service, db.schema, VIDOREV2_ECONOMICS_REPORTS_CONFIG)
            verifier.verify_all()
