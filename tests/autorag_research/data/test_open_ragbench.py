"""Tests for OpenRAGBenchIngestor.

Unit tests for helper functions.
Integration tests use real data subsets against PostgreSQL.
"""

import pytest

from autorag_research.data.open_ragbench import (
    OpenRAGBenchIngestor,
    make_caption_id,
    make_chunk_id,
    make_image_chunk_id,
    make_page_id,
)
from autorag_research.orm.service.multi_modal_ingestion import MultiModalIngestionService
from tests.autorag_research.data.ingestor_test_utils import (
    IngestorTestConfig,
    IngestorTestVerifier,
    create_test_database,
)

# ==================== Unit Tests: Helper Functions ====================


class TestMakePageId:
    def test_make_page_id_basic(self):
        result = make_page_id("2401.12345", 0)
        assert result == "2401.12345_page_0"

    def test_make_page_id_with_version(self):
        result = make_page_id("2401.12345v2", 5)
        assert result == "2401.12345v2_page_5"

    def test_make_page_id_section_index_large(self):
        result = make_page_id("2309.00001", 42)
        assert result == "2309.00001_page_42"


class TestMakeCaptionId:
    def test_make_caption_id_basic(self):
        result = make_caption_id("2401.12345", 0)
        assert result == "2401.12345_caption_0"

    def test_make_caption_id_with_version(self):
        result = make_caption_id("2401.12345v2", 5)
        assert result == "2401.12345v2_caption_5"

    def test_make_caption_id_section_index_large(self):
        result = make_caption_id("2309.00001", 42)
        assert result == "2309.00001_caption_42"


class TestMakeChunkId:
    def test_make_chunk_id_basic(self):
        result = make_chunk_id("2401.12345", 0)
        assert result == "2401.12345_section_0"

    def test_make_chunk_id_with_version(self):
        result = make_chunk_id("2401.12345v2", 5)
        assert result == "2401.12345v2_section_5"

    def test_make_chunk_id_section_index_large(self):
        result = make_chunk_id("2309.00001", 42)
        assert result == "2309.00001_section_42"


class TestMakeImageChunkId:
    def test_make_image_chunk_id_basic(self):
        result = make_image_chunk_id("2401.12345", 0, "img_0")
        assert result == "2401.12345_section_0_img_img_0"

    def test_make_image_chunk_id_different_image_key(self):
        result = make_image_chunk_id("2401.12345", 3, "figure_1")
        assert result == "2401.12345_section_3_img_figure_1"

    def test_make_image_chunk_id_numeric_image_key(self):
        result = make_image_chunk_id("2309.00001", 1, "2")
        assert result == "2309.00001_section_1_img_2"



# ==================== Integration Tests ====================


OPENRAGBENCH_CONFIG = IngestorTestConfig(
    expected_query_count=10,
    expected_chunk_count=10,  # Each query maps to exactly one section
    expected_image_chunk_count=5,  # Some sections have images (figures/tables)
    chunk_count_is_minimum=True,  # Gold sections always included, may have more
    # Full hierarchy checks:
    #   - Chunk: Document -> Page -> Caption -> Chunk
    #   - ImageChunk: Document -> Page -> ImageChunk
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
