"""Tests for MrTyDiIngestor.

Integration tests use real Mr. TyDi dataset subsets against PostgreSQL.
"""

import pytest
from langchain_core.embeddings.fake import FakeEmbeddings

from autorag_research.data.mrtydi import MrTyDiIngestor
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


# ==================== Unit Tests ====================


class TestMrTyDiIngestorInit:
    def test_detect_primary_key_type(self, mock_embedding_model):
        """Mr. TyDi uses string primary keys (e.g., '26569#0')."""
        ingestor = MrTyDiIngestor(mock_embedding_model, language="english")
        assert ingestor.detect_primary_key_type() == "string"


# ==================== Integration Tests ====================


MRTYDI_SMALL_CONFIG = IngestorTestConfig(
    expected_query_count=10,
    expected_chunk_count=50,
    chunk_count_is_minimum=True,  # Gold IDs always included, actual count may exceed limit
    check_retrieval_relations=True,
    check_generation_gt=False,  # Mr. TyDi is retrieval-only, no generation GT
    primary_key_type="string",
    db_name="mrtydi_english_test",
)


@pytest.mark.data
class TestMrTyDiIngestorIntegration:
    """Integration tests using real Mr. TyDi dataset subsets."""

    def test_ingest_english_subset(self, mock_embedding_model):
        """Test ingestion of English Mr. TyDi dataset with query and corpus limits."""
        with create_test_database(MRTYDI_SMALL_CONFIG) as db:
            service = TextDataIngestionService(db.session_factory, schema=db.schema)

            ingestor = MrTyDiIngestor(mock_embedding_model, language="english")
            ingestor.set_service(service)
            ingestor.ingest(
                subset="test",
                query_limit=MRTYDI_SMALL_CONFIG.expected_query_count,
                min_corpus_cnt=MRTYDI_SMALL_CONFIG.expected_chunk_count,
            )

            verifier = IngestorTestVerifier(service, db.schema, MRTYDI_SMALL_CONFIG)
            verifier.verify_all()

    def test_embed_all(self, mock_embedding_model):
        """Test embedding after ingestion."""
        config = IngestorTestConfig(
            expected_query_count=5,
            expected_chunk_count=25,
            chunk_count_is_minimum=True,
            check_retrieval_relations=True,
            check_generation_gt=False,
            primary_key_type="string",
            db_name="mrtydi_embed_test",
        )

        with create_test_database(config) as db:
            service = TextDataIngestionService(db.session_factory, schema=db.schema)

            ingestor = MrTyDiIngestor(mock_embedding_model, language="english")
            ingestor.set_service(service)
            ingestor.ingest(
                subset="test",
                query_limit=config.expected_query_count,
                min_corpus_cnt=config.expected_chunk_count,
            )

            # Before embedding
            stats = service.get_statistics()
            assert stats["chunks"]["with_embeddings"] == 0

            # Run embedding
            ingestor.embed_all(max_concurrency=4, batch_size=10)

            # After embedding
            stats = service.get_statistics()
            assert stats["chunks"]["with_embeddings"] == stats["chunks"]["total"]
            assert stats["chunks"]["without_embeddings"] == 0


class FakeMrTyDiService:
    def __init__(self):
        self.chunk_batches: list[list[dict[str, str | int | None]]] = []

    def add_chunks(self, chunks: list[dict[str, str | int | None]]) -> None:
        self.chunk_batches.append(chunks)


def test_streaming_corpus_ingestion_batches_full_corpus_without_materializing(mock_embedding_model):
    ingestor = MrTyDiIngestor(mock_embedding_model, language="english", batch_size=2)
    service = FakeMrTyDiService()
    ingestor.set_service(service)  # ty: ignore[invalid-argument-type]
    corpus_rows = ({"docid": f"doc-{index}", "title": f"Title {index}", "text": f"Text {index}"} for index in range(5))

    corpus_ids = ingestor._ingest_corpus_streaming(
        corpus_rows,
        gold_docids={"doc-1", "doc-3"},
        min_corpus_cnt=None,
        rng=__import__("random").Random(42),
    )

    assert corpus_ids == {"doc-1", "doc-3"}
    assert [len(batch) for batch in service.chunk_batches] == [2, 2, 1]
    assert service.chunk_batches[0][0] == {"id": "doc-0", "contents": "Title 0 Text 0"}


def test_streaming_corpus_ingestion_reservoir_samples_non_gold_with_limit(mock_embedding_model):
    ingestor = MrTyDiIngestor(mock_embedding_model, language="english", batch_size=2)
    service = FakeMrTyDiService()
    ingestor.set_service(service)  # ty: ignore[invalid-argument-type]
    corpus_rows = ({"docid": f"doc-{index}", "title": f"Title {index}", "text": f"Text {index}"} for index in range(10))

    corpus_ids = ingestor._ingest_corpus_streaming(
        corpus_rows,
        gold_docids={"doc-1", "doc-8"},
        min_corpus_cnt=5,
        rng=__import__("random").Random(42),
    )

    inserted_ids = {str(chunk["id"]) for batch in service.chunk_batches for chunk in batch}
    assert {"doc-1", "doc-8"}.issubset(inserted_ids)
    assert len(inserted_ids) == 5
    assert corpus_ids == inserted_ids
    assert all(len(batch) <= 2 for batch in service.chunk_batches)


def test_query_processing_uses_bounded_reservoir_when_limit_is_set(mock_embedding_model):
    ingestor = MrTyDiIngestor(mock_embedding_model, language="english")
    query_rows = (
        {
            "query_id": index,
            "query": f"query {index}",
            "positive_passages": [{"docid": f"doc-{index}"}],
        }
        for index in range(20)
    )

    queries, qrels, gold_docids = ingestor._process_queries(
        query_rows,
        query_limit=4,
        rng=__import__("random").Random(42),
    )

    assert len(queries) == 4
    assert set(qrels) == set(queries)
    assert len(gold_docids) == 4
