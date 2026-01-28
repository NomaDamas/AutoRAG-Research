"""Tests for ViDoReV3Ingestor.

Integration tests use real data subsets against PostgreSQL.

ViDoReV3 key features:
- Full document hierarchy: File -> Document -> Page -> Caption -> Chunk, ImageChunk
- Query type determines retrieval semantics:
  - multi-hop queries: and_all() semantics (ALL pages required)
  - other queries: or_all() semantics (ANY page acceptable)
- Each page's markdown content becomes one text chunk (no chunking)
"""

import pytest

from autorag_research.data.vidorev3 import ViDoReV3Ingestor
from autorag_research.orm.service.multi_modal_ingestion import MultiModalIngestionService
from tests.autorag_research.data.ingestor_test_utils import (
    IngestorTestConfig,
    IngestorTestVerifier,
    create_test_database,
)

# ==================== Integration Tests ====================


@pytest.mark.data
class TestViDoReV3IngestorIntegration:
    """Integration tests using real ViDoReV3 dataset."""

    @pytest.mark.parametrize(
        "qrels_mode",
        ["image-only", "text-only", "both"],
    )
    def test_ingest_hr_subset(self, qrels_mode: str):
        """Basic integration test - verify_all() handles all standard checks.

        Tests the 'hr' (Human Resources) configuration with a small subset.
        Each page's markdown = one text chunk (no chunking).
        """
        VIDOREV3_HR_CONFIG = IngestorTestConfig(
            expected_query_count=10,
            expected_image_chunk_count=50,  # ImageChunks from corpus pages
            expected_chunk_count=50,  # One text chunk per page (no chunking)
            chunk_count_is_minimum=True,  # Gold IDs always included
            check_files=True,
            expected_file_count=1,
            check_documents=True,
            expected_document_count=1,
            check_pages=True,
            expected_page_count=50,
            check_captions=True,
            expected_caption_count=50,
            check_retrieval_relations=True,
            check_generation_gt=True,
            generation_gt_required_for_all=True,  # All queries have answers in ViDoReV3
            primary_key_type="bigint",  # corpus_id is int64
            db_name=f"vidorev3_hr_test_{qrels_mode.replace('-', '_')}",
        )
        with create_test_database(VIDOREV3_HR_CONFIG) as db:
            service = MultiModalIngestionService(db.session_factory, schema=db.schema)

            ingestor = ViDoReV3Ingestor(
                config_name="hr",
                qrels_mode=qrels_mode,
            )
            ingestor.set_service(service)
            ingestor.ingest(
                query_limit=VIDOREV3_HR_CONFIG.expected_query_count,
                min_corpus_cnt=VIDOREV3_HR_CONFIG.expected_image_chunk_count,
            )

            verifier = IngestorTestVerifier(service, db.schema, VIDOREV3_HR_CONFIG)
            verifier.verify_all()
