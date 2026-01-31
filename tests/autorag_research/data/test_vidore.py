"""Tests for ViDoRe Unified Ingestor (TDD - written BEFORE implementation).

This test file follows the Test-Driven Development approach:
1. Tests are written based on the Mapping_Strategy.md design document
2. Tests define the expected behavior of ViDoReIngestor
3. Implementation will be written to make these tests pass

The unified ViDoReIngestor handles all 10 V1 datasets through a single
parameterized class with dataset_name: VIDORE_V1_DATASETS.
"""

import pytest

from autorag_research.data.vidore import ViDoReIngestor
from autorag_research.orm.service.multi_modal_ingestion import MultiModalIngestionService
from tests.autorag_research.data.ingestor_test_utils import (
    IngestorTestConfig,
    IngestorTestVerifier,
    create_test_database,
)

# ==================== Integration Tests ====================

VIDORE_CONFIG = IngestorTestConfig(
    expected_query_count=10,
    expected_image_chunk_count=10,  # 1:1 query-to-image mapping
    check_retrieval_relations=True,
    check_generation_gt=True,  # arxivqa has answers
    generation_gt_required_for_all=True,  # All arxivqa rows have answers
    primary_key_type="string",
    db_name="vidore_test",
)


@pytest.mark.data
class TestViDoReIngestorIntegration:
    """Integration tests for unified ViDoReIngestor using arxivqa dataset."""

    def test_ingest_arxivqa_subset(self) -> None:
        """Basic integration test - verify_all() handles all standard checks.

        Tests against arxivqa_test_subsampled which has:
        - 1:1 query-image mapping
        - Single letter answers (A/B/C/D)
        - Multiple choice options in query
        """
        with create_test_database(VIDORE_CONFIG) as db:
            service = MultiModalIngestionService(db.session_factory, schema=db.schema)

            ingestor = ViDoReIngestor(dataset_name="arxivqa_test_subsampled")
            ingestor.set_service(service)
            ingestor.ingest(query_limit=VIDORE_CONFIG.expected_query_count)

            verifier = IngestorTestVerifier(service, db.schema, VIDORE_CONFIG)
            verifier.verify_all()

    def test_arxivqa_query_format_contains_options(self) -> None:
        """Test arxivqa-specific query format includes options.

        ArxivQA queries must be formatted as:
        'Given the following query and options, select the correct option.

        Query: <original query>

        Options: <option1>
        <option2>
        ...'
        """
        config = IngestorTestConfig(
            expected_query_count=5,
            expected_image_chunk_count=5,
            check_retrieval_relations=True,
            check_generation_gt=True,
            generation_gt_required_for_all=True,
            primary_key_type="string",
            db_name="vidore_arxivqa_format_test",
        )

        with create_test_database(config) as db:
            service = MultiModalIngestionService(db.session_factory, schema=db.schema)

            ingestor = ViDoReIngestor(dataset_name="arxivqa_test_subsampled")
            ingestor.set_service(service)
            ingestor.ingest(query_limit=config.expected_query_count)

            # Verify ArxivQA-specific query format
            with service._create_uow() as uow:
                queries = uow.queries.get_all(limit=10)
                for query in queries:
                    # Check query contains expected format elements
                    assert "Given the following query and options" in query.contents, (
                        f"Query {query.id} missing header: {query.contents[:100]}..."
                    )
                    assert "Query:" in query.contents, (
                        f"Query {query.id} missing 'Query:' label: {query.contents[:100]}..."
                    )
                    assert "Options:" in query.contents, (
                        f"Query {query.id} missing 'Options:' label: {query.contents[:100]}..."
                    )
