"""Tests for ArxivQA Dataset Ingestor (VisRAG filtered version).

Unit tests for helper functions.
Integration tests use real data subsets against PostgreSQL.

The VisRAG-filtered dataset has:
- corpus: 8,070 images from arXiv papers
- queries: 816 filtered questions (context-independent)
- qrels: Query-to-corpus relevance judgments
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


# VisRAG ArxivQA has BEIR-style format:
# - 816 queries total
# - 8,070 corpus images
# - When using query_limit=10 with corpus_limit=100:
#   - 10 queries sampled
#   - 10 gold images (1:1 in this dataset) + up to 90 additional images = up to 100 total
ARXIVQA_INTEGRATION_CONFIG = IngestorTestConfig(
    expected_query_count=10,
    expected_image_chunk_count=50,  # 10 gold + 40 sampled
    chunk_count_is_minimum=True,  # At least this many (gold IDs always included)
    check_retrieval_relations=True,
    check_generation_gt=True,  # ArxivQA has answers
    generation_gt_required_for_all=True,  # All queries must have generation_gt
    primary_key_type="string",  # VisRAG uses string IDs
    db_name="arxivqa_integration_test",
)


@pytest.mark.data
class TestArxivQAIngestorIntegration:
    """Integration tests using real VisRAG-filtered ArxivQA dataset subsets."""

    def test_ingest_arxivqa_subset(self):
        with create_test_database(ARXIVQA_INTEGRATION_CONFIG) as db:
            service = MultiModalIngestionService(db.session_factory, schema=db.schema)

            ingestor = ArxivQAIngestor()
            ingestor.set_service(service)
            ingestor.ingest(
                query_limit=ARXIVQA_INTEGRATION_CONFIG.expected_query_count,
                corpus_limit=ARXIVQA_INTEGRATION_CONFIG.expected_image_chunk_count,
            )

            verifier = IngestorTestVerifier(service, db.schema, ARXIVQA_INTEGRATION_CONFIG)
            verifier.verify_all()

    def test_query_contents_format(self):
        """Test that query contents include expected format (ArxivQA-specific)."""
        config = IngestorTestConfig(
            expected_query_count=3,
            expected_image_chunk_count=20,  # 3 gold + 17 sampled
            chunk_count_is_minimum=True,
            check_retrieval_relations=True,
            check_generation_gt=True,
            generation_gt_required_for_all=True,
            primary_key_type="string",
            db_name="arxivqa_query_format_test",
        )

        with create_test_database(config) as db:
            service = MultiModalIngestionService(db.session_factory, schema=db.schema)

            ingestor = ArxivQAIngestor()
            ingestor.set_service(service)
            ingestor.ingest(
                query_limit=config.expected_query_count,
                corpus_limit=config.expected_image_chunk_count,
            )

            # Verify ArxivQA-specific query format
            with service._create_uow() as uow:
                queries = uow.queries.get_all(limit=10)
                for query in queries:
                    # Check query contains expected format elements
                    assert "Query:" in query.contents
                    assert "Options:" in query.contents
                    assert "Given the following query and options" in query.contents

    def test_corpus_always_includes_gold_ids(self):
        """Test that corpus filtering always includes gold IDs from qrels."""
        config = IngestorTestConfig(
            expected_query_count=5,
            expected_image_chunk_count=5,  # Minimal: exactly gold IDs
            chunk_count_is_minimum=True,  # Must have at least the gold IDs
            check_retrieval_relations=True,
            check_generation_gt=True,
            generation_gt_required_for_all=True,
            primary_key_type="string",
            db_name="arxivqa_gold_ids_test",
        )

        with create_test_database(config) as db:
            service = MultiModalIngestionService(db.session_factory, schema=db.schema)

            ingestor = ArxivQAIngestor()
            ingestor.set_service(service)
            # Set corpus_limit equal to query_limit to test gold ID inclusion
            ingestor.ingest(
                query_limit=config.expected_query_count,
                corpus_limit=config.expected_query_count,  # Forces minimal corpus
            )

            # Verify all queries have valid retrieval relations
            with service._create_uow() as uow:
                queries = uow.queries.get_all(limit=100)
                relations = uow.retrieval_relations.get_all(limit=1000)

                # All queries should have at least one retrieval relation
                query_ids_with_relations = {r.query_id for r in relations}
                for query in queries:
                    assert query.id in query_ids_with_relations, f"Query {query.id} has no retrieval relations"
