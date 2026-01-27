"""Tests for ViDoReV3Ingestor.

Unit tests for helper functions.
Integration tests use real data subsets against PostgreSQL.

ViDoReV3 key features:
- Full document hierarchy: File -> Document -> Page -> Caption -> Chunk, ImageChunk
- Query type determines retrieval semantics:
  - multi-hop queries: and_all() semantics (ALL pages required)
  - other queries: or_all() semantics (ANY page acceptable)
- LlamaIndex chunking for Caption -> Chunk splitting
"""

import pytest
from llama_index.core.node_parser import SentenceSplitter
from omegaconf import DictConfig

from autorag_research.data.vidorev3 import (
    ViDoReV3Ingestor,
    _resolve_chunker,
    chunk_markdown,
)
from autorag_research.orm.service.multi_modal_ingestion import MultiModalIngestionService
from tests.autorag_research.data.ingestor_test_utils import (
    IngestorTestConfig,
    IngestorTestVerifier,
    create_test_database,
)

# ==================== Unit Tests ====================


class TestResolveChunker:
    """Unit tests for _resolve_chunker helper function."""

    def test_resolve_chunker_with_none_returns_default(self):
        """When chunker is None, return default SentenceSplitter."""
        chunker = _resolve_chunker(None)
        assert chunker is None

    def test_resolve_chunker_with_node_parser_returns_same(self):
        """When chunker is a NodeParser, return it unchanged."""
        custom_chunker = SentenceSplitter(chunk_size=256, chunk_overlap=32)
        result = _resolve_chunker(custom_chunker)
        assert result is custom_chunker

    def test_resolve_chunker_with_dict_config_instantiates(self):
        """When chunker is a DictConfig, instantiate it via Hydra."""
        config = DictConfig({
            "_target_": "llama_index.core.node_parser.SentenceSplitter",
            "chunk_size": 512,
            "chunk_overlap": 64,
        })
        chunker = _resolve_chunker(config)
        assert isinstance(chunker, SentenceSplitter)
        assert chunker.chunk_size == 512
        assert chunker.chunk_overlap == 64


class TestChunkMarkdown:
    """Unit tests for chunk_markdown function."""

    def test_chunk_markdown_empty_string_returns_empty_list(self):
        """Empty markdown returns empty list."""
        chunker = SentenceSplitter(chunk_size=100, chunk_overlap=10)
        assert chunk_markdown("", chunker) == []

    def test_chunk_markdown_whitespace_only_returns_empty_list(self):
        """Whitespace-only markdown returns empty list."""
        chunker = SentenceSplitter(chunk_size=100, chunk_overlap=10)
        assert chunk_markdown("   \n\t  ", chunker) == []

    def test_chunk_markdown_with_content_returns_chunks(self):
        """Non-empty markdown returns list of chunk strings."""
        chunker = SentenceSplitter(chunk_size=100, chunk_overlap=10)
        result = chunk_markdown("This is some test content.", chunker)
        assert len(result) > 0
        assert all(isinstance(c, str) for c in result)


# ==================== Integration Tests ====================

VIDOREV3_HR_CONFIG = IngestorTestConfig(
    expected_query_count=10,
    expected_image_chunk_count=50,  # ImageChunks from corpus pages
    expected_chunk_count=50,  # Text chunks from Caption via LlamaIndex chunking (minimum, varies by text length)
    chunk_count_is_minimum=True,  # Gold IDs always included, chunk count varies by text length.
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
    db_name="vidorev3_hr_test",
)


@pytest.mark.data
class TestViDoReV3IngestorIntegration:
    """Integration tests using real ViDoReV3 dataset."""

    def test_ingest_hr_subset(self):
        """Basic integration test - verify_all() handles all standard checks.

        Tests the 'hr' (Human Resources) configuration with a small subset.
        """
        with create_test_database(VIDOREV3_HR_CONFIG) as db:
            service = MultiModalIngestionService(db.session_factory, schema=db.schema)

            ingestor = ViDoReV3Ingestor(config_name="hr")
            ingestor.set_service(service)
            ingestor.ingest(
                query_limit=VIDOREV3_HR_CONFIG.expected_query_count,
                min_corpus_cnt=VIDOREV3_HR_CONFIG.expected_image_chunk_count,
            )

            verifier = IngestorTestVerifier(service, db.schema, VIDOREV3_HR_CONFIG)
            verifier.verify_all()
