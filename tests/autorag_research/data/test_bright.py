"""Unit tests for BRIGHTIngestor.

Tests cover:
- Constructor validation (domain, document_mode)
- Domain selection logic
- Document mode selection
- ID prefixing/transformation
- Corpus, query, and relation ingestion flows
- Edge cases and error handling
"""

from unittest.mock import MagicMock, patch

import pytest
from llama_index.core import MockEmbedding

from autorag_research.data.bright import (
    BRIGHT_DOMAINS,
    DOMAINS_WITH_LONG_DOCS,
    DOMAINS_WITHOUT_LONG_DOCS,
    BRIGHTIngestor,
    _get_gold_ids,
    _make_chunk_id,
    _make_query_id,
    _process_gold_answer,
)
from autorag_research.exceptions import ServiceNotSetError

EMBEDDING_DIM = 768


# ==================== Fixtures ====================


@pytest.fixture
def mock_embedding_model():
    return MockEmbedding(EMBEDDING_DIM)


@pytest.fixture
def mock_service():
    service = MagicMock()
    service.add_chunks = MagicMock(return_value=[1, 2, 3])
    service.add_queries = MagicMock(return_value=[1, 2])
    service.add_retrieval_gt = MagicMock(return_value=[(1, 0, 0)])
    service.clean = MagicMock(return_value={"deleted_queries": 0, "deleted_chunks": 0})
    service.embed_all_queries = MagicMock(return_value=10)
    service.embed_all_chunks = MagicMock(return_value=100)
    return service


@pytest.fixture
def sample_corpus_data():
    return [
        {"id": "doc_001", "content": "This is the first document content."},
        {"id": "doc_002", "content": "This is the second document content."},
        {"id": "doc_003", "content": "This is the third document content."},
    ]


@pytest.fixture
def sample_examples_data():
    return [
        {
            "id": "q_001",
            "query": "What is the first document about?",
            "gold_ids": ["doc_001", "doc_002"],
            "gold_ids_long": ["long_doc_001"],
            "gold_answer": "The first document is about testing.",
            "reasoning": "Based on the content provided.",
            "excluded_ids": ["doc_003"],
        },
        {
            "id": "q_002",
            "query": "What is the second document about?",
            "gold_ids": ["doc_002"],
            "gold_ids_long": ["long_doc_002", "N/A"],
            "gold_answer": "N/A",
            "reasoning": "empty",
            "excluded_ids": ["N/A"],
        },
    ]


def create_mock_dataset(data: list[dict]):
    yield from data


# ==================== Constructor Validation Tests ====================


class TestBRIGHTIngestorInit:
    def test_init_default_domains(self, mock_embedding_model):
        ingestor = BRIGHTIngestor(mock_embedding_model)
        assert ingestor.domains == BRIGHT_DOMAINS
        assert ingestor.document_mode == "short"

    def test_init_custom_domains(self, mock_embedding_model):
        custom_domains = ["biology", "economics"]
        ingestor = BRIGHTIngestor(mock_embedding_model, domains=custom_domains)
        assert ingestor.domains == custom_domains

    def test_init_custom_document_mode_long(self, mock_embedding_model):
        ingestor = BRIGHTIngestor(
            mock_embedding_model,
            domains=["biology"],
            document_mode="long",
        )
        assert ingestor.document_mode == "long"

    def test_init_invalid_domain_raises_error(self, mock_embedding_model):
        with pytest.raises(ValueError, match="Invalid domain 'invalid_domain'"):
            BRIGHTIngestor(mock_embedding_model, domains=["invalid_domain"])

    def test_init_long_mode_with_unsupported_domain_raises_error(self, mock_embedding_model):
        with pytest.raises(ValueError, match="does not support long documents"):
            BRIGHTIngestor(mock_embedding_model, domains=["leetcode"], document_mode="long")

    def test_init_long_mode_with_all_unsupported_domains_raises_error(self, mock_embedding_model):
        for domain in DOMAINS_WITHOUT_LONG_DOCS:
            with pytest.raises(ValueError, match="does not support long documents"):
                BRIGHTIngestor(mock_embedding_model, domains=[domain], document_mode="long")

    def test_init_long_mode_with_supported_domains_succeeds(self, mock_embedding_model):
        for domain in DOMAINS_WITH_LONG_DOCS:
            ingestor = BRIGHTIngestor(mock_embedding_model, domains=[domain], document_mode="long")
            assert ingestor.document_mode == "long"


class TestBRIGHTIngestorValidateDomains:
    def test_validate_all_bright_domains_valid(self, mock_embedding_model):
        ingestor = BRIGHTIngestor(mock_embedding_model)
        assert all(d in BRIGHT_DOMAINS for d in ingestor.domains)

    def test_validate_mixed_valid_invalid_domains_raises_error(self, mock_embedding_model):
        with pytest.raises(ValueError, match="Invalid domain"):
            BRIGHTIngestor(mock_embedding_model, domains=["biology", "not_a_domain"])


# ==================== Primary Key Type Tests ====================


class TestBRIGHTIngestorDetectPrimaryKeyType:
    def test_detect_primary_key_type_returns_string(self, mock_embedding_model):
        ingestor = BRIGHTIngestor(mock_embedding_model)
        assert ingestor.detect_primary_key_type() == "string"


# ==================== Helper Function Tests ====================


class TestMakeQueryId:
    def test_make_query_id_basic(self):
        result = _make_query_id("biology", "123")
        assert result == "biology_123"

    def test_make_query_id_with_complex_id(self):
        result = _make_query_id("stackoverflow", "question_456_abc")
        assert result == "stackoverflow_question_456_abc"


class TestMakeChunkId:
    def test_make_chunk_id_basic(self):
        result = _make_chunk_id("economics", "doc_789")
        assert result == "economics_doc_789"

    def test_make_chunk_id_with_numeric_source_id(self):
        result = _make_chunk_id("robotics", "12345")
        assert result == "robotics_12345"


class TestProcessGoldAnswer:
    def test_process_gold_answer_with_valid_answer(self):
        result = _process_gold_answer("This is a valid answer.")
        assert result == ["This is a valid answer."]

    def test_process_gold_answer_with_na(self):
        result = _process_gold_answer("N/A")
        assert result is None

    def test_process_gold_answer_empty_string(self):
        result = _process_gold_answer("")
        assert result == [""]


class TestGetGoldIds:
    def test_get_gold_ids_short_mode(self):
        example = {"gold_ids": ["id1", "id2"], "gold_ids_long": ["long_id1"]}
        result = _get_gold_ids(example, "short", "biology")
        assert result == ["id1", "id2"]

    def test_get_gold_ids_long_mode(self):
        example = {"gold_ids": ["id1", "id2"], "gold_ids_long": ["long_id1", "long_id2"]}
        result = _get_gold_ids(example, "long", "biology")
        assert result == ["long_id1", "long_id2"]

    def test_get_gold_ids_long_mode_filters_na(self):
        example = {"gold_ids": ["id1"], "gold_ids_long": ["long_id1", "N/A", "long_id2"]}
        result = _get_gold_ids(example, "long", "biology")
        assert result == ["long_id1", "long_id2"]

    def test_get_gold_ids_long_mode_all_na_returns_empty(self):
        example = {"gold_ids": ["id1"], "gold_ids_long": ["N/A"]}
        result = _get_gold_ids(example, "long", "biology")
        assert result == []

    def test_get_gold_ids_long_mode_unsupported_domain_raises_error(self):
        example = {"gold_ids": ["id1"], "gold_ids_long": ["long_id1"]}
        with pytest.raises(ValueError, match="does not have long documents"):
            _get_gold_ids(example, "long", "leetcode")


# ==================== Ingest Flow Tests ====================


class TestBRIGHTIngestorIngest:
    @patch("autorag_research.data.bright.load_dataset")
    def test_ingest_raises_error_without_service(self, mock_load_dataset, mock_embedding_model):
        ingestor = BRIGHTIngestor(mock_embedding_model, domains=["biology"])
        with pytest.raises(ServiceNotSetError):
            ingestor.ingest()

    @patch("autorag_research.data.bright.load_dataset")
    def test_ingest_calls_service_methods(
        self, mock_load_dataset, mock_embedding_model, mock_service, sample_corpus_data, sample_examples_data
    ):
        mock_load_dataset.side_effect = [
            create_mock_dataset(sample_corpus_data),
            create_mock_dataset(sample_examples_data),
        ]

        ingestor = BRIGHTIngestor(mock_embedding_model, domains=["biology"])
        ingestor.set_service(mock_service)
        ingestor.ingest()

        mock_service.add_chunks.assert_called()
        mock_service.add_queries.assert_called()
        mock_service.add_retrieval_gt.assert_called()
        mock_service.clean.assert_called_once()

    @patch("autorag_research.data.bright.load_dataset")
    def test_ingest_processes_multiple_domains(
        self, mock_load_dataset, mock_embedding_model, mock_service, sample_corpus_data, sample_examples_data
    ):
        mock_load_dataset.side_effect = [
            create_mock_dataset(sample_corpus_data),
            create_mock_dataset(sample_examples_data),
            create_mock_dataset(sample_corpus_data),
            create_mock_dataset(sample_examples_data),
        ]

        ingestor = BRIGHTIngestor(mock_embedding_model, domains=["biology", "economics"])
        ingestor.set_service(mock_service)
        ingestor.ingest()

        assert mock_load_dataset.call_count == 4
        assert mock_service.add_chunks.call_count == 2
        assert mock_service.add_queries.call_count == 2


class TestBRIGHTIngestorIngestCorpus:
    @patch("autorag_research.data.bright.load_dataset")
    def test_ingest_corpus_raises_error_without_service(self, mock_load_dataset, mock_embedding_model):
        ingestor = BRIGHTIngestor(mock_embedding_model, domains=["biology"])
        with pytest.raises(ServiceNotSetError):
            ingestor._ingest_corpus("biology")

    @patch("autorag_research.data.bright.load_dataset")
    def test_ingest_corpus_short_mode_uses_documents_config(
        self, mock_load_dataset, mock_embedding_model, mock_service, sample_corpus_data
    ):
        mock_load_dataset.return_value = create_mock_dataset(sample_corpus_data)

        ingestor = BRIGHTIngestor(mock_embedding_model, domains=["biology"], document_mode="short")
        ingestor.set_service(mock_service)
        ingestor._ingest_corpus("biology")

        mock_load_dataset.assert_called_once_with(
            "xlangai/BRIGHT", "documents", split="biology", streaming=True, trust_remote_code=True
        )

    @patch("autorag_research.data.bright.load_dataset")
    def test_ingest_corpus_long_mode_uses_long_documents_config(
        self, mock_load_dataset, mock_embedding_model, mock_service, sample_corpus_data
    ):
        mock_load_dataset.return_value = create_mock_dataset(sample_corpus_data)

        ingestor = BRIGHTIngestor(mock_embedding_model, domains=["biology"], document_mode="long")
        ingestor.set_service(mock_service)
        ingestor._ingest_corpus("biology")

        mock_load_dataset.assert_called_once_with(
            "xlangai/BRIGHT", "long_documents", split="biology", streaming=True, trust_remote_code=True
        )

    @patch("autorag_research.data.bright.load_dataset")
    def test_ingest_corpus_creates_prefixed_ids(
        self, mock_load_dataset, mock_embedding_model, mock_service, sample_corpus_data
    ):
        mock_load_dataset.return_value = create_mock_dataset(sample_corpus_data)

        ingestor = BRIGHTIngestor(mock_embedding_model, domains=["biology"])
        ingestor.set_service(mock_service)
        ingestor._ingest_corpus("biology")

        call_args = mock_service.add_chunks.call_args[0][0]
        expected_ids = ["biology_doc_001", "biology_doc_002", "biology_doc_003"]
        actual_ids = [chunk["id"] for chunk in call_args]
        assert actual_ids == expected_ids

    @patch("autorag_research.data.bright.load_dataset")
    def test_ingest_corpus_returns_total_count(
        self, mock_load_dataset, mock_embedding_model, mock_service, sample_corpus_data
    ):
        mock_load_dataset.return_value = create_mock_dataset(sample_corpus_data)

        ingestor = BRIGHTIngestor(mock_embedding_model, domains=["biology"])
        ingestor.set_service(mock_service)
        count = ingestor._ingest_corpus("biology")

        assert count == 3

    @patch("autorag_research.data.bright.load_dataset")
    def test_ingest_corpus_empty_dataset_returns_zero(self, mock_load_dataset, mock_embedding_model, mock_service):
        mock_load_dataset.return_value = create_mock_dataset([])

        ingestor = BRIGHTIngestor(mock_embedding_model, domains=["biology"])
        ingestor.set_service(mock_service)
        count = ingestor._ingest_corpus("biology")

        assert count == 0
        mock_service.add_chunks.assert_not_called()


class TestBRIGHTIngestorIngestQueries:
    @patch("autorag_research.data.bright.load_dataset")
    def test_ingest_queries_raises_error_without_service(self, mock_load_dataset, mock_embedding_model):
        ingestor = BRIGHTIngestor(mock_embedding_model, domains=["biology"])
        with pytest.raises(ServiceNotSetError):
            ingestor._ingest_queries("biology")

    @patch("autorag_research.data.bright.load_dataset")
    def test_ingest_queries_creates_prefixed_ids(
        self, mock_load_dataset, mock_embedding_model, mock_service, sample_examples_data
    ):
        mock_load_dataset.return_value = create_mock_dataset(sample_examples_data)

        ingestor = BRIGHTIngestor(mock_embedding_model, domains=["biology"])
        ingestor.set_service(mock_service)
        query_ids, _ = ingestor._ingest_queries("biology")

        assert query_ids == ["biology_q_001", "biology_q_002"]

    @patch("autorag_research.data.bright.load_dataset")
    def test_ingest_queries_returns_prefixed_gold_ids(
        self, mock_load_dataset, mock_embedding_model, mock_service, sample_examples_data
    ):
        mock_load_dataset.return_value = create_mock_dataset(sample_examples_data)

        ingestor = BRIGHTIngestor(mock_embedding_model, domains=["biology"])
        ingestor.set_service(mock_service)
        _, gold_ids_list = ingestor._ingest_queries("biology")

        assert gold_ids_list[0] == ["biology_doc_001", "biology_doc_002"]
        assert gold_ids_list[1] == ["biology_doc_002"]

    @patch("autorag_research.data.bright.load_dataset")
    def test_ingest_queries_processes_gold_answer_na(
        self, mock_load_dataset, mock_embedding_model, mock_service, sample_examples_data
    ):
        mock_load_dataset.return_value = create_mock_dataset(sample_examples_data)

        ingestor = BRIGHTIngestor(mock_embedding_model, domains=["biology"])
        ingestor.set_service(mock_service)
        ingestor._ingest_queries("biology")

        call_args = mock_service.add_queries.call_args[0][0]
        assert call_args[0]["generation_gt"] == ["The first document is about testing."]
        assert call_args[1]["generation_gt"] is None

    @patch("autorag_research.data.bright.load_dataset")
    def test_ingest_queries_skips_queries_without_gold_ids(self, mock_load_dataset, mock_embedding_model, mock_service):
        data_with_empty_gold = [
            {
                "id": "q_001",
                "query": "Query with no gold ids",
                "gold_ids": [],
                "gold_ids_long": [],
                "gold_answer": "Some answer",
                "reasoning": "Some reasoning",
                "excluded_ids": [],
            }
        ]
        mock_load_dataset.return_value = create_mock_dataset(data_with_empty_gold)

        ingestor = BRIGHTIngestor(mock_embedding_model, domains=["biology"])
        ingestor.set_service(mock_service)
        query_ids, gold_ids_list = ingestor._ingest_queries("biology")

        assert query_ids == []
        assert gold_ids_list == []

    @patch("autorag_research.data.bright.load_dataset")
    def test_ingest_queries_long_mode_filters_na_gold_ids(
        self, mock_load_dataset, mock_embedding_model, mock_service, sample_examples_data
    ):
        mock_load_dataset.return_value = create_mock_dataset(sample_examples_data)

        ingestor = BRIGHTIngestor(mock_embedding_model, domains=["biology"], document_mode="long")
        ingestor.set_service(mock_service)
        query_ids, gold_ids_list = ingestor._ingest_queries("biology")

        assert len(query_ids) == 2
        assert gold_ids_list[0] == ["biology_long_doc_001"]
        assert gold_ids_list[1] == ["biology_long_doc_002"]


class TestBRIGHTIngestorIngestRelations:
    def test_ingest_relations_raises_error_without_service(self, mock_embedding_model):
        ingestor = BRIGHTIngestor(mock_embedding_model, domains=["biology"])
        with pytest.raises(ServiceNotSetError):
            ingestor._ingest_relations(["q1"], [["c1", "c2"]])

    def test_ingest_relations_calls_service_for_each_query(self, mock_embedding_model, mock_service):
        ingestor = BRIGHTIngestor(mock_embedding_model, domains=["biology"])
        ingestor.set_service(mock_service)

        query_ids = ["biology_q_001", "biology_q_002"]
        gold_ids_list = [["biology_doc_001", "biology_doc_002"], ["biology_doc_003"]]
        ingestor._ingest_relations(query_ids, gold_ids_list)

        assert mock_service.add_retrieval_gt.call_count == 2

    def test_ingest_relations_skips_empty_gold_ids(self, mock_embedding_model, mock_service):
        ingestor = BRIGHTIngestor(mock_embedding_model, domains=["biology"])
        ingestor.set_service(mock_service)

        query_ids = ["biology_q_001", "biology_q_002", "biology_q_003"]
        gold_ids_list = [["biology_doc_001"], [], ["biology_doc_002"]]
        ingestor._ingest_relations(query_ids, gold_ids_list)

        assert mock_service.add_retrieval_gt.call_count == 2


# ==================== Embed All Tests ====================


class TestBRIGHTIngestorEmbedAll:
    def test_embed_all_raises_error_without_service(self, mock_embedding_model):
        ingestor = BRIGHTIngestor(mock_embedding_model, domains=["biology"])
        with pytest.raises(ServiceNotSetError):
            ingestor.embed_all()

    def test_embed_all_calls_embed_methods(self, mock_embedding_model, mock_service):
        ingestor = BRIGHTIngestor(mock_embedding_model, domains=["biology"])
        ingestor.set_service(mock_service)
        ingestor.embed_all(max_concurrency=8, batch_size=64)

        mock_service.embed_all_queries.assert_called_once()
        mock_service.embed_all_chunks.assert_called_once()

        query_call_kwargs = mock_service.embed_all_queries.call_args
        assert query_call_kwargs.kwargs["batch_size"] == 64
        assert query_call_kwargs.kwargs["max_concurrency"] == 8


# ==================== Set Service Tests ====================


class TestBRIGHTIngestorSetService:
    def test_set_service_stores_service(self, mock_embedding_model, mock_service):
        ingestor = BRIGHTIngestor(mock_embedding_model, domains=["biology"])
        assert ingestor.service is None

        ingestor.set_service(mock_service)
        assert ingestor.service is mock_service


# ==================== Integration-Style Tests (Mocked Dependencies) ====================


class TestBRIGHTIngestorIntegration:
    @patch("autorag_research.data.bright.load_dataset")
    def test_full_ingest_flow_short_mode(
        self, mock_load_dataset, mock_embedding_model, mock_service, sample_corpus_data, sample_examples_data
    ):
        mock_load_dataset.side_effect = [
            create_mock_dataset(sample_corpus_data),
            create_mock_dataset(sample_examples_data),
        ]

        ingestor = BRIGHTIngestor(mock_embedding_model, domains=["biology"], document_mode="short")
        ingestor.set_service(mock_service)
        ingestor.ingest()

        chunks_call = mock_service.add_chunks.call_args[0][0]
        assert len(chunks_call) == 3
        assert all("biology_" in chunk["id"] for chunk in chunks_call)

        queries_call = mock_service.add_queries.call_args[0][0]
        assert len(queries_call) == 2
        assert queries_call[0]["contents"] == "What is the first document about?"

    @patch("autorag_research.data.bright.load_dataset")
    def test_full_ingest_flow_long_mode(
        self, mock_load_dataset, mock_embedding_model, mock_service, sample_corpus_data, sample_examples_data
    ):
        mock_load_dataset.side_effect = [
            create_mock_dataset(sample_corpus_data),
            create_mock_dataset(sample_examples_data),
        ]

        ingestor = BRIGHTIngestor(mock_embedding_model, domains=["biology"], document_mode="long")
        ingestor.set_service(mock_service)
        ingestor.ingest()

        mock_load_dataset.assert_any_call(
            "xlangai/BRIGHT", "long_documents", split="biology", streaming=True, trust_remote_code=True
        )

    @patch("autorag_research.data.bright.load_dataset")
    def test_ingest_with_batch_processing(self, mock_load_dataset, mock_embedding_model, mock_service):
        large_corpus = [{"id": f"doc_{i:04d}", "content": f"Content {i}"} for i in range(2500)]
        large_examples = [
            {
                "id": f"q_{i:04d}",
                "query": f"Query {i}",
                "gold_ids": [f"doc_{i:04d}"],
                "gold_ids_long": [f"long_doc_{i:04d}"],
                "gold_answer": f"Answer {i}",
                "reasoning": "Some reasoning",
                "excluded_ids": [],
            }
            for i in range(100)
        ]

        mock_load_dataset.side_effect = [
            create_mock_dataset(large_corpus),
            create_mock_dataset(large_examples),
        ]

        ingestor = BRIGHTIngestor(mock_embedding_model, domains=["biology"])
        ingestor.set_service(mock_service)
        ingestor.ingest()

        assert mock_service.add_chunks.call_count == 3
