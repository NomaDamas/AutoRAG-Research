"""Tests for CRAGIngestor."""

import bz2
import importlib
import io
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.embeddings.fake import FakeEmbeddings

from autorag_research.data.crag import (
    CRAGIngestor,
    _build_generation_gt,
    _format_search_result_contents,
    _iter_crag_examples,
    _make_chunk_id,
    _make_query_id,
    _resolve_subset,
)
from autorag_research.data.registry import get_ingestor
from autorag_research.orm.service.text_ingestion import TextDataIngestionService
from tests.autorag_research.data.ingestor_test_utils import (
    IngestorTestConfig,
    create_test_database,
)

EMBEDDING_DIM = 768

SAMPLE_EXAMPLES = [
    {
        "interaction_id": "dev-1",
        "query": "Who won the championship?",
        "answer": "The Tigers",
        "alt_ans": ["Tigers", "The Tigers"],
        "split": 0,
        "search_results": [
            {
                "page_name": "Championship recap",
                "page_url": "https://example.com/recap",
                "page_snippet": "The Tigers won the final.",
                "page_result": "<html><body><h1>Recap</h1><p>The Tigers won the final.</p></body></html>",
                "page_last_modified": "Mon, 11 Mar 2024 06:06:35 GMT",
            },
            {
                "page_name": "Box score",
                "page_url": "https://example.com/box",
                "page_snippet": "Box score summary",
                "page_result": "",
                "page_last_modified": "",
            },
        ],
    },
    {
        "interaction_id": "test-1",
        "query": "Who directed the movie?",
        "answer": "Jane Doe",
        "alt_ans": [],
        "split": 1,
        "search_results": [
            {
                "page_name": "Movie profile",
                "page_url": "https://example.com/movie",
                "page_snippet": "Jane Doe directed the movie.",
                "page_result": "<html><body><p>Jane Doe directed the movie.</p></body></html>",
                "page_last_modified": "Tue, 12 Mar 2024 06:06:35 GMT",
            }
        ],
    },
]


@pytest.fixture
def mock_embedding_model():
    return FakeEmbeddings(size=EMBEDDING_DIM)


class TestResolveSubset:
    def test_resolve_supported_subsets(self):
        assert _resolve_subset("train") == 0
        assert _resolve_subset("dev") == 0
        assert _resolve_subset("test") == 1


class TestBuildGenerationGt:
    def test_build_generation_gt_merges_primary_and_alternates_without_duplicates(self):
        assert _build_generation_gt("The Tigers", ["Tigers", "The Tigers", "  "]) == ["The Tigers", "Tigers"]

    def test_build_generation_gt_returns_none_when_all_answers_empty(self):
        assert _build_generation_gt("", [" ", ""]) is None


class TestChunkFormatting:
    def test_format_search_result_contents_includes_metadata_and_html_text(self):
        result = _format_search_result_contents(SAMPLE_EXAMPLES[0]["search_results"][0])
        assert "Title: Championship recap" in result
        assert "URL: https://example.com/recap" in result
        assert "Snippet: The Tigers won the final." in result
        assert "Last Modified: Mon, 11 Mar 2024 06:06:35 GMT" in result
        assert "Recap" in result
        assert "The Tigers won the final." in result
        assert "<html>" not in result

    def test_format_search_result_contents_falls_back_to_snippet_without_html(self):
        result = _format_search_result_contents(SAMPLE_EXAMPLES[0]["search_results"][1])
        assert "Snippet: Box score summary" in result
        assert "Content:" in result
        assert "Box score summary" in result


class TestIdsAndRegistry:
    def test_make_query_id(self):
        assert _make_query_id("dev", "abc-123") == "crag_dev_abc-123"

    def test_make_chunk_id(self):
        assert _make_chunk_id("dev", "abc-123", 4) == "crag_dev_abc-123_4"

    def test_registry_metadata_exposes_crag_ingestor(self):
        import autorag_research.data.crag as crag_module

        importlib.reload(crag_module)

        meta = get_ingestor("crag")
        assert meta is not None
        assert meta.name == "crag"
        assert meta.hf_repo is None


class TestCRAGStreaming:
    @patch("autorag_research.data.crag.urllib.request.urlopen")
    def test_streaming_download_uses_timeout(self, mock_urlopen):
        mock_urlopen.return_value = io.BytesIO(bz2.compress(b'{"interaction_id": "dev-1"}\n'))

        assert list(_iter_crag_examples()) == [{"interaction_id": "dev-1"}]
        mock_urlopen.assert_called_once_with(
            "https://github.com/facebookresearch/CRAG/raw/refs/heads/main/data/crag_task_1_and_2_dev_v4.jsonl.bz2",
            timeout=60,
        )

    @patch("autorag_research.data.crag.urllib.request.urlopen")
    def test_streaming_rejects_oversized_json_record(self, mock_urlopen, monkeypatch):
        monkeypatch.setattr("autorag_research.data.crag.CRAG_MAX_JSON_LINE_BYTES", 32, raising=False)
        mock_urlopen.return_value = io.BytesIO(bz2.compress(b'{"value": "' + b"x" * 64 + b'"}\n'))

        with pytest.raises(ValueError, match="CRAG JSONL record exceeds"):
            list(_iter_crag_examples())


class TestIngestUnit:
    @patch("autorag_research.data.crag._iter_crag_examples")
    def test_ingest_filters_to_requested_split_and_skips_retrieval_gt(self, mock_iter_examples, mock_embedding_model):
        mock_iter_examples.return_value = iter(SAMPLE_EXAMPLES)
        service = MagicMock()
        ingestor = CRAGIngestor(mock_embedding_model, batch_size=10)
        ingestor.set_service(service)

        ingestor.ingest(subset="dev", query_limit=1)

        service.add_queries.assert_called_once_with([
            {
                "id": "crag_dev_dev-1",
                "contents": "Who won the championship?",
                "generation_gt": ["The Tigers", "Tigers"],
            }
        ])
        service.add_chunks.assert_called_once_with([
            {
                "id": "crag_dev_dev-1_0",
                "contents": _format_search_result_contents(SAMPLE_EXAMPLES[0]["search_results"][0]),
            },
            {
                "id": "crag_dev_dev-1_1",
                "contents": _format_search_result_contents(SAMPLE_EXAMPLES[0]["search_results"][1]),
            },
        ])
        service.add_retrieval_gt.assert_not_called()
        service.clean.assert_called_once_with()

    @patch("autorag_research.data.crag._iter_crag_examples")
    def test_ingest_warns_that_train_alias_duplicates_dev_examples(
        self,
        mock_iter_examples,
        mock_embedding_model,
        caplog,
    ):
        mock_iter_examples.return_value = iter(())
        service = MagicMock()
        ingestor = CRAGIngestor(mock_embedding_model, batch_size=10)
        ingestor.set_service(service)

        with caplog.at_level("WARNING", logger="AutoRAG-Research"):
            ingestor.ingest(subset="train")

        assert "Ingesting both subset='train' and subset='dev' will duplicate the same upstream examples" in caplog.text


CRAG_INTEGRATION_CONFIG = IngestorTestConfig(
    expected_query_count=2,
    expected_chunk_count=10,
    check_retrieval_relations=False,
    check_generation_gt=True,
    generation_gt_required_for_all=True,
    primary_key_type="string",
    db_name="crag_integration_test",
)


@pytest.mark.data
class TestCRAGIngestorIntegration:
    def test_ingest_dev_subset(self, mock_embedding_model):
        with create_test_database(CRAG_INTEGRATION_CONFIG) as db:
            service = TextDataIngestionService(db.session_factory, schema=db.schema)
            ingestor = CRAGIngestor(mock_embedding_model, batch_size=2)
            ingestor.set_service(service)

            ingestor.ingest(subset="dev", query_limit=2)

            stats = service.get_statistics()
            assert stats["queries"]["total"] == 2
            assert stats["chunks"]["total"] == 10

            with service._create_uow() as uow:
                queries = uow.queries.get_all(limit=None)
                chunks = uow.chunks.get_all(limit=10)

            assert all(query.generation_gt for query in queries)
            assert any(chunk.contents.startswith("Title:") for chunk in chunks)
            assert any("Content:" in chunk.contents for chunk in chunks)
