"""Tests for ArxivQA (Full 100K) Dataset Ingestor.

Unit tests for helper functions (TDD - written before implementation).
Integration tests use real data subsets against PostgreSQL.
"""

import io

import pytest
from PIL import Image

from autorag_research.data.arxivqa import ArxivQAIngestor
from autorag_research.orm.service.multi_modal_ingestion import MultiModalIngestionService
from tests.autorag_research.data.ingestor_test_utils import (
    IngestorTestConfig,
    IngestorTestVerifier,
    create_test_database,
)

# ==================== Unit Tests: Label Normalization ====================


class TestNormalizeLabel:
    """Test _normalize_label static method for various label formats."""

    @pytest.mark.parametrize(
        "label,expected",
        [
            ("A", "A"),
            ("B", "B"),
            ("C", "C"),
            ("D", "D"),
            ("E", "E"),
            ("a", "A"),
            ("b", "B"),
            ("c", "C"),
        ],
    )
    def test_single_letter(self, label: str, expected: str):
        options = ["A. First", "B. Second", "C. Third", "D. Fourth"]
        result = ArxivQAIngestor._normalize_label(label, options)
        assert result == expected

    @pytest.mark.parametrize(
        "label,expected",
        [
            ("A.", "A"),
            ("B.", "B"),
            ("A)", "A"),
            ("B)", "B"),
            ("C. Some text here", "C"),
            ("D) Another option", "D"),
            ("e.", "E"),
        ],
    )
    def test_letter_with_separator(self, label: str, expected: str):
        options = ["A. First", "B. Second", "C. Third", "D. Fourth", "E. Fifth"]
        result = ArxivQAIngestor._normalize_label(label, options)
        assert result == expected

    def test_full_option_text_matching(self):
        options = ["A. Apple", "B. Banana", "C. Cherry", "D. Date"]
        # Label is the full option text without prefix
        result = ArxivQAIngestor._normalize_label("Banana", options)
        assert result == "B"

    def test_full_option_with_prefix(self):
        options = ["A. Apple", "B. Banana", "C. Cherry", "D. Date"]
        result = ArxivQAIngestor._normalize_label("B. Banana", options)
        assert result == "B"

    def test_case_insensitive_option_matching(self):
        options = ["A. APPLE", "B. Banana", "C. cherry", "D. Date"]
        result = ArxivQAIngestor._normalize_label("apple", options)
        assert result == "A"

    def test_whitespace_handling(self):
        options = ["A. First", "B. Second"]
        result = ArxivQAIngestor._normalize_label("  A  ", options)
        assert result == "A"


# ==================== Unit Tests: Query Formatting ====================


class TestFormatQuery:
    """Test _format_query static method."""

    def test_format_query_basic(self):
        question = "What color is the sky?"
        options = ["A. Red", "B. Blue", "C. Green", "D. Yellow"]
        result = ArxivQAIngestor._format_query(question, options)

        assert "Given the following query and options" in result
        assert "Query: What color is the sky?" in result
        assert "Options:" in result
        assert "A. Red" in result
        assert "B. Blue" in result
        assert "C. Green" in result
        assert "D. Yellow" in result

    def test_format_query_with_five_options(self):
        question = "Sample question?"
        options = ["A. One", "B. Two", "C. Three", "D. Four", "E. Five"]
        result = ArxivQAIngestor._format_query(question, options)

        assert "E. Five" in result

    def test_format_query_options_joined_with_newlines(self):
        question = "Question?"
        options = ["A. First", "B. Second"]
        result = ArxivQAIngestor._format_query(question, options)

        # Options should be on separate lines
        assert "A. First\nB. Second" in result


# ==================== Unit Tests: PIL to Bytes Conversion ====================


class TestPilToBytes:
    """Test _pil_to_bytes static method."""

    def test_jpeg_conversion_rgb(self):
        img = Image.new("RGB", (100, 100), color="red")
        img_bytes, mimetype = ArxivQAIngestor._pil_to_bytes(img)

        assert isinstance(img_bytes, bytes)
        assert len(img_bytes) > 0
        assert mimetype == "image/jpeg"

        # Verify bytes can be read back as image
        loaded_img = Image.open(io.BytesIO(img_bytes))
        assert loaded_img.size == (100, 100)

    def test_jpeg_conversion_grayscale(self):
        img = Image.new("L", (50, 50), color=128)
        img_bytes, mimetype = ArxivQAIngestor._pil_to_bytes(img)

        assert isinstance(img_bytes, bytes)
        assert mimetype == "image/jpeg"

    def test_png_conversion_rgba(self):
        img = Image.new("RGBA", (50, 50), color=(255, 0, 0, 128))
        img_bytes, mimetype = ArxivQAIngestor._pil_to_bytes(img)

        assert isinstance(img_bytes, bytes)
        assert mimetype == "image/png"

    def test_png_conversion_la_mode(self):
        img = Image.new("LA", (30, 30), color=(128, 200))
        img_bytes, mimetype = ArxivQAIngestor._pil_to_bytes(img)

        assert isinstance(img_bytes, bytes)
        assert mimetype == "image/png"

    def test_png_conversion_palette_mode(self):
        img = Image.new("RGB", (20, 20), color="blue")
        img_p = img.convert("P")
        img_bytes, mimetype = ArxivQAIngestor._pil_to_bytes(img_p)

        assert isinstance(img_bytes, bytes)
        assert mimetype == "image/png"


# ==================== Integration Tests ====================


ARXIVQA_INTEGRATION_CONFIG = IngestorTestConfig(
    expected_query_count=10,
    expected_image_chunk_count=10,  # 1:1 query to image mapping
    chunk_count_is_minimum=False,  # Exact counts for ArxivQA
    check_retrieval_relations=True,
    check_generation_gt=True,  # ArxivQA has answers
    generation_gt_required_for_all=True,  # All queries must have generation_gt
    primary_key_type="bigint",
    db_name="arxivqa_integration_test",
)


@pytest.mark.data
class TestArxivQAIngestorIntegration:
    """Integration tests using real ArxivQA dataset subsets."""

    def test_ingest_arxivqa_subset(self):
        with create_test_database(ARXIVQA_INTEGRATION_CONFIG) as db:
            service = MultiModalIngestionService(db.session_factory, schema=db.schema)

            ingestor = ArxivQAIngestor()
            ingestor.set_service(service)
            ingestor.ingest(
                query_limit=ARXIVQA_INTEGRATION_CONFIG.expected_query_count,
            )

            verifier = IngestorTestVerifier(service, db.schema, ARXIVQA_INTEGRATION_CONFIG)
            verifier.verify_all()

    def test_query_contents_format(self):
        """Test that query contents include expected format (ArxivQA-specific)."""
        config = IngestorTestConfig(
            expected_query_count=3,
            expected_image_chunk_count=3,
            chunk_count_is_minimum=False,
            check_retrieval_relations=True,
            check_generation_gt=True,
            generation_gt_required_for_all=True,
            primary_key_type="bigint",
            db_name="arxivqa_query_format_test",
        )

        with create_test_database(config) as db:
            service = MultiModalIngestionService(db.session_factory, schema=db.schema)

            ingestor = ArxivQAIngestor()
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
