"""Tests for SDS KoPub VDR ingestor."""

from __future__ import annotations

import importlib
from typing import Any

import pandas as pd
import pytest
from PIL import Image

import autorag_research.data.sds_kopub_vdr as sds_kopub_vdr_module
from autorag_research.data.registry import discover_ingestors
from autorag_research.orm.models.retrieval_gt import gt_to_relations, normalize_gt
from autorag_research.orm.service.multi_modal_ingestion import MultiModalIngestionService
from tests.autorag_research.data.ingestor_test_utils import (
    IngestorTestConfig,
    IngestorTestVerifier,
    create_test_database,
)


class _FakeDataset:
    def __init__(self, rows: list[dict[str, Any]]):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows)


class _FakeService:
    def __init__(self):
        self.files: list[dict[str, Any]] = []
        self.documents: list[dict[str, Any]] = []
        self.pages: list[dict[str, Any]] = []
        self.chunks: list[dict[str, Any]] = []
        self.image_chunks: list[dict[str, Any]] = []
        self.queries: list[dict[str, Any]] = []
        self.page_chunk_relations: list[tuple[str, str]] = []
        self.retrieval_gt_items: list[tuple[str, Any]] = []
        self.retrieval_gt_chunk_type: str | None = None

    def add_files(self, files: list[dict[str, Any]]) -> list[str]:
        ids = [f"file-{len(self.files) + idx}" for idx in range(len(files))]
        self.files.extend(files)
        return ids

    def add_documents(self, documents: list[dict[str, Any]]) -> list[str]:
        ids = [f"doc-{len(self.documents) + idx}" for idx in range(len(documents))]
        self.documents.extend(documents)
        return ids

    def add_pages(self, pages: list[dict[str, Any]]) -> list[str]:
        ids = [f"page-{len(self.pages) + idx}" for idx in range(len(pages))]
        self.pages.extend(pages)
        return ids

    def add_chunks(self, chunks: list[dict[str, Any]]) -> list[str]:
        self.chunks.extend(chunks)
        return [str(chunk["id"]) for chunk in chunks]

    def link_page_to_chunks(self, page_id: str, chunk_ids: list[str]) -> list[tuple[str, str]]:
        relations = [(page_id, chunk_id) for chunk_id in chunk_ids]
        self.page_chunk_relations.extend(relations)
        return relations

    def add_image_chunks(self, image_chunks: list[dict[str, Any]]) -> list[str]:
        self.image_chunks.extend(image_chunks)
        return [str(chunk["id"]) for chunk in image_chunks]

    def add_queries(self, queries: list[dict[str, Any]]) -> list[str]:
        self.queries.extend(queries)
        return [str(query["id"]) for query in queries]

    def add_retrieval_gt_batch(
        self,
        items: list[tuple[str, Any]],
        chunk_type: str = "mixed",
    ) -> list[tuple[str, int, int]]:
        self.retrieval_gt_items.extend(items)
        self.retrieval_gt_chunk_type = chunk_type
        return []


@pytest.fixture
def fake_sds_rows() -> dict[tuple[str, str], list[dict[str, Any]]]:
    image = Image.new("RGB", (2, 2), color="white")
    return {
        ("queries", "test"): [
            {"id": "query-0", "text": "첫 번째 질문"},
            {"id": "query-1", "text": "두 번째 질문"},
        ],
        ("qrels", "test"): [
            {"query-id": "query-0", "corpus-id": "doc-a_1", "score": 1},
            {"query-id": "query-1", "corpus-id": "doc-b_1", "score": 1},
            {"query-id": "query-1", "corpus-id": "doc-b_2", "score": 0},
        ],
        ("corpus", "test"): [
            {"id": "doc-a_0", "text": "추가 문서", "image": image},
            {"id": "doc-a_1", "text": "첫 번째 근거", "image": image},
            {"id": "doc-b_1", "text": "두 번째 근거", "image": image},
        ],
    }


def test_sds_kopub_vdr_detects_string_primary_key():
    ingestor = sds_kopub_vdr_module.SDSKoPubVDRIngestor()

    assert ingestor.detect_primary_key_type() == "string"


def test_sds_kopub_vdr_is_registered_with_qrels_mode_choices():
    discover_ingestors.cache_clear()
    # Some registry tests clear the global registry after this module has already
    # been imported. Reload to re-run the @register_ingestor decorator so this
    # test is order-independent in the full suite.
    importlib.reload(sds_kopub_vdr_module)
    meta = discover_ingestors()["sds_kopub_vdr"]

    assert meta.ingestor_class is sds_kopub_vdr_module.SDSKoPubVDRIngestor
    qrels_param = next(param for param in meta.params if param.name == "qrels_mode")
    assert qrels_param.choices == ["image", "text", "mixed"]


def test_sds_kopub_vdr_ingests_mteb_rows(monkeypatch, fake_sds_rows):
    def fake_load_dataset(_path: str, config: str, streaming: bool, split: str):
        assert streaming is (config == "corpus")
        return _FakeDataset(fake_sds_rows[(config, split)])

    monkeypatch.setattr("autorag_research.data.sds_kopub_vdr.load_dataset", fake_load_dataset)
    service = _FakeService()
    ingestor = sds_kopub_vdr_module.SDSKoPubVDRIngestor()
    ingestor.set_service(service)  # type: ignore[arg-type]

    ingestor.ingest(min_corpus_cnt=3)

    assert [query["id"] for query in service.queries] == ["query-0", "query-1"]
    assert "generation_gt" not in service.queries[0]
    assert {chunk["id"] for chunk in service.image_chunks} == {"doc-a_0", "doc-a_1", "doc-b_1"}
    assert {chunk["id"] for chunk in service.chunks} == {"doc-a_0", "doc-a_1", "doc-b_1"}
    assert service.retrieval_gt_chunk_type == "image"
    assert len(service.documents) == 2

    q0_gt = dict(service.retrieval_gt_items)["query-0"]
    q0_relations = gt_to_relations("query-0", normalize_gt(q0_gt, chunk_type="image"))
    assert [relation["image_chunk_id"] for relation in q0_relations] == ["doc-a_1"]
    assert [relation["score"] for relation in q0_relations] == [1]


def test_sds_kopub_vdr_infer_page_num_warns_on_non_numeric_suffix(caplog):
    ingestor = sds_kopub_vdr_module.SDSKoPubVDRIngestor()

    with caplog.at_level("WARNING", logger="AutoRAG-Research"):
        page_num = ingestor._infer_page_num("malformed_corpus_id_abc")

    assert page_num == 1
    assert any("non-numeric trailing segment" in record.message for record in caplog.records)


def test_sds_kopub_vdr_infer_page_num_silent_on_numeric_suffix(caplog):
    ingestor = sds_kopub_vdr_module.SDSKoPubVDRIngestor()

    with caplog.at_level("WARNING", logger="AutoRAG-Research"):
        page_num = ingestor._infer_page_num("public_pdf/foo/bar_42")

    assert page_num == 42
    assert not any("non-numeric trailing segment" in record.message for record in caplog.records)


def test_sds_kopub_vdr_supports_mixed_qrels(monkeypatch, fake_sds_rows):
    def fake_load_dataset(_path: str, config: str, streaming: bool, split: str):
        return _FakeDataset(fake_sds_rows[(config, split)])

    monkeypatch.setattr("autorag_research.data.sds_kopub_vdr.load_dataset", fake_load_dataset)
    service = _FakeService()
    ingestor = sds_kopub_vdr_module.SDSKoPubVDRIngestor(qrels_mode="mixed")
    ingestor.set_service(service)  # type: ignore[arg-type]

    ingestor.ingest(query_limit=1)

    assert service.retrieval_gt_chunk_type == "mixed"
    query_id, gt = service.retrieval_gt_items[0]
    relations = gt_to_relations(query_id, normalize_gt(gt, chunk_type="mixed"))
    assert query_id == "query-0"
    assert {relation["image_chunk_id"] for relation in relations} == {"doc-a_1", None}
    assert {relation["chunk_id"] for relation in relations} == {"doc-a_1", None}


# With seed=42 and query_limit=5, the SDS KoPub VDR ingestor selects 5 queries
# whose gold corpus IDs cover 4 distinct source documents and 5 distinct pages
# (one document — ``public_pdf/prism/2040 아산시 환경계획(최종안)`` — contributes
# 2 of the 5 gold pages: page 330 for ``query-281`` and page 379 for ``query-250``).
# ``min_corpus_cnt=20`` then pads the corpus to ≥20 image+text chunks; in real ingest
# the padding tends to pull in additional source documents (the @pytest.mark.data run
# observes ~18 distinct documents/files). All count fields below are therefore
# interpreted as minimums via ``chunk_count_is_minimum=True`` rather than equalities.
SDS_KOPUB_VDR_CONFIG = IngestorTestConfig(
    expected_query_count=5,
    expected_chunk_count=20,
    expected_image_chunk_count=20,
    chunk_count_is_minimum=True,
    check_documents=True,
    expected_document_count=5,
    check_pages=True,
    expected_page_count=20,
    check_files=True,
    expected_file_count=5,
    check_retrieval_relations=True,
    check_generation_gt=False,
    check_relevance_scores=True,
    primary_key_type="string",
    db_name="sds_kopub_vdr_test",
)


@pytest.mark.data
class TestSDSKoPubVDRIngestorIntegration:
    """Integration tests using the real SDS KoPub VDR MTEB dataset."""

    def test_ingest_subset(self):
        with create_test_database(SDS_KOPUB_VDR_CONFIG) as db:
            service = MultiModalIngestionService(db.session_factory, schema=db.schema)  # ty: ignore[invalid-argument-type]

            ingestor = sds_kopub_vdr_module.SDSKoPubVDRIngestor()
            ingestor.set_service(service)
            ingestor.ingest(
                query_limit=SDS_KOPUB_VDR_CONFIG.expected_query_count,
                min_corpus_cnt=SDS_KOPUB_VDR_CONFIG.expected_image_chunk_count,
            )

            verifier = IngestorTestVerifier(service, db.schema, SDS_KOPUB_VDR_CONFIG)
            verifier.verify_all()
