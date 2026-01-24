"""Tests for ViDoRe Ingestors.

Unit tests for helper functions and constructor validation.
Integration tests use real data subsets against PostgreSQL.
"""

import pytest

from autorag_research.data.vidore import (
    ViDoReArxivQAIngestor,
    ViDoReDatasets,
)
from autorag_research.orm.service.multi_modal_ingestion import MultiModalIngestionService
from tests.autorag_research.data.ingestor_test_utils import (
    IngestorTestConfig,
    IngestorTestVerifier,
    create_test_database,
)

# ==================== Unit Tests: Dataset Validation ====================


class TestViDoReDatasetValidation:
    """Test dataset name validation for ViDoReIngestor."""

    def test_valid_dataset_names(self):
        """Verify all valid dataset names are in the list."""
        expected_datasets = [
            "arxivqa_test_subsampled",
            "docvqa_test_subsampled",
            "infovqa_test_subsampled",
            "tabfquad_test_subsampled",
            "tatdqa_test",
            "shiftproject_test",
            "syntheticDocQA_artificial_intelligence_test",
            "syntheticDocQA_energy_test",
            "syntheticDocQA_government_reports_test",
            "syntheticDocQA_healthcare_industry_test",
        ]
        assert ViDoReDatasets == expected_datasets


# ==================== Integration Tests ====================


VIDORE_INTEGRATION_CONFIG = IngestorTestConfig(
    expected_query_count=10,
    expected_image_chunk_count=10,  # ViDoRe has 1:1 query to image mapping
    chunk_count_is_minimum=False,  # Exact counts for ViDoRe
    check_retrieval_relations=True,
    check_generation_gt=True,  # ViDoRe arxivqa has answers
    generation_gt_required_for_all=True,  # All queries must have generation_gt
    primary_key_type="bigint",
    db_name="vidore_integration_test",
)


@pytest.mark.data
class TestViDoReArxivQAIngestorIntegration:
    """Integration tests using real ViDoRe arxivqa dataset subsets."""

    def test_ingest_arxivqa_subset(self):
        """Test ingestion of arxivqa dataset with query limit."""
        with create_test_database(VIDORE_INTEGRATION_CONFIG) as db:
            service = MultiModalIngestionService(db.session_factory, schema=db.schema)

            ingestor = ViDoReArxivQAIngestor()
            ingestor.set_service(service)
            ingestor.ingest(
                query_limit=VIDORE_INTEGRATION_CONFIG.expected_query_count,
            )

            verifier = IngestorTestVerifier(service, db.schema, VIDORE_INTEGRATION_CONFIG)
            verifier.verify_all()

    def test_query_contents_format(self):
        """Test that query contents include options in expected format (ArxivQA-specific)."""
        config = IngestorTestConfig(
            expected_query_count=3,
            expected_image_chunk_count=3,
            chunk_count_is_minimum=False,
            check_retrieval_relations=True,
            check_generation_gt=True,
            generation_gt_required_for_all=True,
            primary_key_type="bigint",
            db_name="vidore_query_format_test",
        )

        with create_test_database(config) as db:
            service = MultiModalIngestionService(db.session_factory, schema=db.schema)

            ingestor = ViDoReArxivQAIngestor()
            ingestor.set_service(service)
            ingestor.ingest(query_limit=config.expected_query_count)

            # Verify ArxivQA-specific query format
            with service._create_uow() as uow:
                queries = uow.queries.get_all(limit=10)
                for query in queries:
                    # Check query contains expected format elements
                    assert "Query:" in query.contents
                    assert "Options:" in query.contents
                    assert "Given the following query and options" in query.contents
