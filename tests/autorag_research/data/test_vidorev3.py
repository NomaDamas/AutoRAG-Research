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
        assert isinstance(chunker, SentenceSplitter)

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
    expected_chunk_count=10,  # Text chunks from Caption via LlamaIndex chunking (minimum, varies by text length)
    chunk_count_is_minimum=True,  # Gold IDs always included, chunk count varies by text length
    # Full hierarchy checks
    # Note: Document/File counts vary based on which queries are sampled
    # and their associated corpus IDs. Disabled exact matching as counts depend on data.
    check_files=False,
    expected_file_count=None,
    check_documents=False,
    expected_document_count=None,
    check_pages=False,  # Page count varies based on corpus sampling
    check_captions=False,  # Caption count varies (some pages may have empty markdown)
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

    def test_multi_hop_query_uses_and_semantics(self):
        """Test that multi-hop queries use AND semantics (group_index increments).

        ViDoReV3-specific business logic: When query_types contains 'multi-hop',
        each relevant corpus_id gets a different group_index (AND semantics),
        meaning ALL pages are required for correct retrieval.
        """
        config = IngestorTestConfig(
            expected_query_count=20,  # Need enough to find multi-hop queries
            expected_image_chunk_count=100,
            chunk_count_is_minimum=True,
            check_retrieval_relations=True,
            check_generation_gt=True,
            generation_gt_required_for_all=True,
            primary_key_type="bigint",
            db_name="vidorev3_multihop_test",
        )

        with create_test_database(config) as db:
            service = MultiModalIngestionService(db.session_factory, schema=db.schema)

            ingestor = ViDoReV3Ingestor(config_name="hr")
            ingestor.set_service(service)
            ingestor.ingest(
                query_limit=config.expected_query_count,
                min_corpus_cnt=config.expected_image_chunk_count,
            )

            # Verify AND semantics for multi-hop queries
            # Multi-hop queries should have relations with different group_index values
            # (each corpus_id is a required hop)
            with service._create_uow() as uow:
                all_queries = uow.queries.get_all(limit=100)

                found_multi_hop = False
                for query in all_queries:
                    relations = uow.retrieval_relations.get_by_query_id(query.id)
                    if len(relations) > 1:
                        # Check if this query uses AND semantics (different group_index)
                        group_indices = {rel.group_index for rel in relations}
                        if len(group_indices) > 1:
                            # This is a multi-hop query with AND semantics
                            found_multi_hop = True
                            # Verify each relation has group_order = 0 (single item per AND group)
                            for rel in relations:
                                assert rel.group_order == 0, (
                                    f"Multi-hop query {query.id}: expected group_order=0 for all relations, "
                                    f"got group_order={rel.group_order} at group_index={rel.group_index}"
                                )
                            # Verify group_indices are sequential from 0
                            expected_indices = set(range(len(relations)))
                            assert group_indices == expected_indices, (
                                f"Multi-hop query {query.id}: expected sequential group_indices {expected_indices}, "
                                f"got {group_indices}"
                            )

                # Note: Not all datasets/configs may have multi-hop queries
                # This test passes if found and validates correctly, or if none exist
                if not found_multi_hop:
                    pytest.skip("No multi-hop queries found in this subset - cannot verify AND semantics")

    def test_non_multi_hop_query_uses_or_semantics(self):
        """Test that non-multi-hop queries use OR semantics (group_order increments).

        ViDoReV3-specific business logic: When query_types does NOT contain 'multi-hop',
        all relevant corpus_ids share group_index=0 with incrementing group_order (OR semantics),
        meaning ANY page is acceptable for correct retrieval.
        """
        config = IngestorTestConfig(
            expected_query_count=10,
            expected_image_chunk_count=50,
            chunk_count_is_minimum=True,
            check_retrieval_relations=True,
            check_generation_gt=True,
            generation_gt_required_for_all=True,
            primary_key_type="bigint",
            db_name="vidorev3_or_semantics_test",
        )

        with create_test_database(config) as db:
            service = MultiModalIngestionService(db.session_factory, schema=db.schema)

            ingestor = ViDoReV3Ingestor(config_name="hr")
            ingestor.set_service(service)
            ingestor.ingest(
                query_limit=config.expected_query_count,
                min_corpus_cnt=config.expected_image_chunk_count,
            )

            # Verify OR semantics for non-multi-hop queries
            with service._create_uow() as uow:
                all_queries = uow.queries.get_all(limit=100)

                found_or_query = False
                for query in all_queries:
                    relations = uow.retrieval_relations.get_by_query_id(query.id)
                    if len(relations) > 1:
                        # Check if this query uses OR semantics (same group_index)
                        group_indices = {rel.group_index for rel in relations}
                        if len(group_indices) == 1 and 0 in group_indices:
                            # This is a non-multi-hop query with OR semantics
                            found_or_query = True
                            # Verify group_order values are sequential from 0
                            group_orders = sorted(rel.group_order for rel in relations)
                            expected_orders = list(range(len(relations)))
                            assert group_orders == expected_orders, (
                                f"OR query {query.id}: expected sequential group_orders {expected_orders}, "
                                f"got {group_orders}"
                            )

                assert found_or_query, (
                    "No non-multi-hop queries with multiple relations found - cannot verify OR semantics"
                )

    def test_full_hierarchy_caption_to_chunk_relationship(self):
        """Test Caption -> Chunk relationship via LlamaIndex chunking.

        ViDoReV3-specific: Caption stores full markdown text, Chunks are created
        by splitting Caption contents using LlamaIndex SentenceSplitter.
        """
        config = IngestorTestConfig(
            expected_query_count=5,
            expected_image_chunk_count=20,
            expected_chunk_count=5,  # At least some chunks from captions
            chunk_count_is_minimum=True,
            check_retrieval_relations=True,
            primary_key_type="bigint",
            db_name="vidorev3_hierarchy_test",
        )

        with create_test_database(config) as db:
            service = MultiModalIngestionService(db.session_factory, schema=db.schema)

            ingestor = ViDoReV3Ingestor(config_name="hr")
            ingestor.set_service(service)
            ingestor.ingest(
                query_limit=config.expected_query_count,
                min_corpus_cnt=config.expected_image_chunk_count,
            )

            with service._create_uow() as uow:
                # Get all chunks and verify they have parent_caption
                chunks = uow.chunks.get_all(limit=100)
                if chunks:
                    for chunk in chunks:
                        assert chunk.parent_caption is not None, (
                            f"Chunk {chunk.id} should have parent_caption FK set"
                        )
                        # Verify the parent caption exists
                        caption = uow.captions.get_by_id(chunk.parent_caption)
                        assert caption is not None, (
                            f"Chunk {chunk.id} references non-existent caption {chunk.parent_caption}"
                        )
                        # Chunk contents should be a substring or derived from caption contents
                        # (LlamaIndex may add some metadata, so we just check non-empty)
                        assert chunk.contents and len(chunk.contents) > 0, (
                            f"Chunk {chunk.id} should have non-empty contents"
                        )

                # Get all captions and verify hierarchy
                captions = uow.captions.get_all(limit=100)
                if captions:
                    for caption in captions:
                        assert caption.page_id is not None, (
                            f"Caption {caption.id} should have page_id FK set"
                        )
                        # Verify parent page exists
                        page = uow.pages.get_by_id(caption.page_id)
                        assert page is not None, (
                            f"Caption {caption.id} references non-existent page {caption.page_id}"
                        )

    def test_image_chunk_has_parent_page(self):
        """Test ImageChunk -> Page relationship.

        ViDoReV3-specific: ImageChunks store page images and must reference their parent Page.
        """
        config = IngestorTestConfig(
            expected_query_count=5,
            expected_image_chunk_count=10,
            chunk_count_is_minimum=True,
            check_retrieval_relations=True,
            primary_key_type="bigint",
            db_name="vidorev3_image_hierarchy_test",
        )

        with create_test_database(config) as db:
            service = MultiModalIngestionService(db.session_factory, schema=db.schema)

            ingestor = ViDoReV3Ingestor(config_name="hr")
            ingestor.set_service(service)
            ingestor.ingest(
                query_limit=config.expected_query_count,
                min_corpus_cnt=config.expected_image_chunk_count,
            )

            with service._create_uow() as uow:
                # Get all image chunks and verify they have parent_page
                image_chunks = uow.image_chunks.get_all(limit=100)
                assert len(image_chunks) > 0, "Should have at least one ImageChunk"

                for img_chunk in image_chunks:
                    assert img_chunk.parent_page is not None, (
                        f"ImageChunk {img_chunk.id} should have parent_page FK set"
                    )
                    # Verify the parent page exists
                    page = uow.pages.get_by_id(img_chunk.parent_page)
                    assert page is not None, (
                        f"ImageChunk {img_chunk.id} references non-existent page {img_chunk.parent_page}"
                    )
