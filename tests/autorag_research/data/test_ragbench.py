"""Tests for RAGBenchIngestor.

Unit tests for helper functions (TDD - tests written before implementation).
Integration tests use real data subsets against PostgreSQL.
"""

import pytest
from langchain_core.embeddings.fake import FakeEmbeddings

from autorag_research.data.ragbench import (
    RAGBenchIngestor,
    _make_query_id,
    compute_chunk_id,
    extract_relevant_doc_indices,
)
from autorag_research.orm.service.text_ingestion import TextDataIngestionService
from tests.autorag_research.data.ingestor_test_utils import (
    IngestorTestConfig,
    IngestorTestVerifier,
    create_test_database,
)

EMBEDDING_DIM = 768


# ==================== Fixtures ====================


@pytest.fixture
def mock_embedding_model():
    return FakeEmbeddings(size=EMBEDDING_DIM)


# ==================== Unit Tests: Helper Functions ====================


class TestExtractRelevantDocIndices:
    def test_extract_single_digit_indices(self):
        sentence_keys = ["0a", "0b", "1a", "2c"]
        result = extract_relevant_doc_indices(sentence_keys)
        assert result == {0, 1, 2}

    def test_extract_multi_digit_indices(self):
        sentence_keys = ["10a", "12b", "0c", "5d"]
        result = extract_relevant_doc_indices(sentence_keys)
        assert result == {0, 5, 10, 12}

    def test_extract_duplicate_doc_indices(self):
        sentence_keys = ["0a", "0b", "0c", "0d", "0e", "0f"]
        result = extract_relevant_doc_indices(sentence_keys)
        assert result == {0}

    def test_extract_empty_list(self):
        result = extract_relevant_doc_indices([])
        assert result == set()

    def test_extract_mixed_indices(self):
        sentence_keys = ["0d", "0e", "0f", "1d", "1e", "1f"]
        result = extract_relevant_doc_indices(sentence_keys)
        assert result == {0, 1}


class TestComputeChunkId:
    def test_compute_chunk_id_basic(self):
        content = "Title: Test Document\nPassage: This is a test passage."
        config = "covidqa"
        result = compute_chunk_id(content, config)
        assert result.startswith("covidqa_")
        assert len(result) == len("covidqa_") + 16  # config + underscore + 16 char hash

    def test_compute_chunk_id_same_content_same_hash(self):
        content = "Title: Same\nPassage: Same content"
        config = "covidqa"
        result1 = compute_chunk_id(content, config)
        result2 = compute_chunk_id(content, config)
        assert result1 == result2

    def test_compute_chunk_id_different_content_different_hash(self):
        config = "covidqa"
        result1 = compute_chunk_id("Content A", config)
        result2 = compute_chunk_id("Content B", config)
        assert result1 != result2

    def test_compute_chunk_id_different_config_different_hash(self):
        content = "Same content"
        result1 = compute_chunk_id(content, "covidqa")
        result2 = compute_chunk_id(content, "hotpotqa")
        assert result1 != result2
        assert result1.startswith("covidqa_")
        assert result2.startswith("hotpotqa_")

    def test_compute_chunk_id_normalization(self):
        config = "covidqa"
        result1 = compute_chunk_id("  Content   with   spaces  ", config)
        result2 = compute_chunk_id("content with spaces", config)
        assert result1 == result2

    def test_compute_chunk_id_case_insensitive(self):
        config = "covidqa"
        result1 = compute_chunk_id("CONTENT", config)
        result2 = compute_chunk_id("content", config)
        assert result1 == result2


class TestMakeQueryId:
    def test_make_query_id_basic(self):
        result = _make_query_id("covidqa", "train", "358")
        assert result == "covidqa_train_358"

    def test_make_query_id_different_split(self):
        result = _make_query_id("hotpotqa", "test", "123")
        assert result == "hotpotqa_test_123"

    def test_make_query_id_validation_split(self):
        result = _make_query_id("finqa", "validation", "456")
        assert result == "finqa_validation_456"


# ==================== Integration Tests ====================


RAGBENCH_INTEGRATION_CONFIG = IngestorTestConfig(
    expected_query_count=10,
    expected_chunk_count=40,  # ~3-4 docs per query, some deduplication
    chunk_count_is_minimum=True,  # Exact count depends on deduplication
    check_retrieval_relations=True,
    check_generation_gt=True,
    generation_gt_required_for_all=True,  # RAGbench has responses for all queries
    primary_key_type="string",
    db_name="ragbench_integration_test",
)


@pytest.mark.data
class TestRAGBenchIngestorIntegration:
    def test_ingest_covidqa_subset(self, mock_embedding_model):
        with create_test_database(RAGBENCH_INTEGRATION_CONFIG) as db:
            service = TextDataIngestionService(db.session_factory, schema=db.schema)

            ingestor = RAGBenchIngestor(
                mock_embedding_model,
                config="covidqa",
            )
            ingestor.set_service(service)
            ingestor.ingest(
                query_limit=RAGBENCH_INTEGRATION_CONFIG.expected_query_count,
                subset="test",
            )

            verifier = IngestorTestVerifier(service, db.schema, RAGBENCH_INTEGRATION_CONFIG)
            verifier.verify_all()


class FakeRAGBenchService:
    def __init__(self):
        self.chunks: list[list[dict[str, str | int | None]]] = []
        self.queries: list[list[dict[str, str | list[str] | None]]] = []
        self.retrieval_gt_calls: list[tuple[str, object, str, bool]] = []

    def add_chunks(self, chunks: list[dict[str, str | int | None]]) -> None:
        self.chunks.append(chunks)

    def add_queries(self, queries: list[dict[str, str | list[str] | None]]) -> None:
        self.queries.append(queries)

    def add_retrieval_gt(self, query_id: str, gt: object, chunk_type: str = "mixed", upsert: bool = False) -> None:
        self.retrieval_gt_calls.append((query_id, gt, chunk_type, upsert))


def _extract_or_gt_ids(gt: object) -> list[str]:
    if hasattr(gt, "items"):
        return [str(item.id) for item in gt.items]
    if hasattr(gt, "value"):
        return [str(gt.value)]
    raise AssertionError


def test_process_batch_upserts_duplicate_ragbench_query_relations_across_batches(mock_embedding_model):
    service = FakeRAGBenchService()
    ingestor = RAGBenchIngestor(mock_embedding_model, config="covidqa", batch_size=1)
    ingestor.set_service(service)  # ty: ignore[invalid-argument-type]
    seen_chunk_ids: set[str] = set()
    relation_chunk_ids_by_query: dict[str, set[str]] = {}

    first_example = {
        "id": "dup",
        "question": "What is duplicated?",
        "response": "First response",
        "documents": ["First supporting document."],
        "all_relevant_sentence_keys": ["0a"],
    }
    second_example = {
        "id": "dup",
        "question": "What is duplicated?",
        "response": "First response",
        "documents": ["Second supporting document.", "Shared supporting document."],
        "all_relevant_sentence_keys": ["0a", "1b"],
    }

    ingestor._process_batch("covidqa", "test", [first_example], seen_chunk_ids, relation_chunk_ids_by_query)
    ingestor._process_batch("covidqa", "test", [second_example], seen_chunk_ids, relation_chunk_ids_by_query)

    query_id = _make_query_id("covidqa", "test", "dup")
    first_chunk_id = compute_chunk_id("First supporting document.", "covidqa")
    second_chunk_id = compute_chunk_id("Second supporting document.", "covidqa")
    shared_chunk_id = compute_chunk_id("Shared supporting document.", "covidqa")

    assert [call[0] for call in service.retrieval_gt_calls] == [query_id, query_id]
    assert [call[2] for call in service.retrieval_gt_calls] == ["text", "text"]
    assert [call[3] for call in service.retrieval_gt_calls] == [True, True]
    assert _extract_or_gt_ids(service.retrieval_gt_calls[0][1]) == [first_chunk_id]
    assert _extract_or_gt_ids(service.retrieval_gt_calls[1][1]) == sorted([
        first_chunk_id,
        second_chunk_id,
        shared_chunk_id,
    ])


def test_process_batch_preserves_duplicate_ragbench_query_relations_within_same_batch(mock_embedding_model):
    service = FakeRAGBenchService()
    ingestor = RAGBenchIngestor(mock_embedding_model, config="covidqa")
    ingestor.set_service(service)  # ty: ignore[invalid-argument-type]
    seen_chunk_ids: set[str] = set()
    relation_chunk_ids_by_query: dict[str, set[str]] = {}

    first_example = {
        "id": "dup",
        "question": "What is duplicated?",
        "response": "Same response",
        "documents": ["First supporting document."],
        "all_relevant_sentence_keys": ["0a"],
    }
    second_example = {
        "id": "dup",
        "question": "What is duplicated?",
        "response": "Same response",
        "documents": ["Second supporting document.", "Shared supporting document."],
        "all_relevant_sentence_keys": ["0a", "1b"],
    }

    ingestor._process_batch(
        "covidqa",
        "test",
        [first_example, second_example],
        seen_chunk_ids,
        relation_chunk_ids_by_query,
    )

    query_id = _make_query_id("covidqa", "test", "dup")
    first_chunk_id = compute_chunk_id("First supporting document.", "covidqa")
    second_chunk_id = compute_chunk_id("Second supporting document.", "covidqa")
    shared_chunk_id = compute_chunk_id("Shared supporting document.", "covidqa")

    assert [call[0] for call in service.retrieval_gt_calls] == [query_id, query_id]
    assert [call[2] for call in service.retrieval_gt_calls] == ["text", "text"]
    assert [call[3] for call in service.retrieval_gt_calls] == [True, True]
    assert _extract_or_gt_ids(service.retrieval_gt_calls[0][1]) == [first_chunk_id]
    assert _extract_or_gt_ids(service.retrieval_gt_calls[1][1]) == sorted([
        first_chunk_id,
        second_chunk_id,
        shared_chunk_id,
    ])
