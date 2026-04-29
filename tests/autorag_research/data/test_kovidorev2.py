"""Tests for KoViDoRe V2 ingestor."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from PIL import Image

from autorag_research.data.kovidorev2 import KoViDoReV2Ingestor
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
        self.retrieval_gt_items: list[tuple[int, Any]] = []
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
        items: list[tuple[int, Any]],
        chunk_type: str = "mixed",
    ) -> list[tuple[int, int, int]]:
        self.retrieval_gt_items.extend(items)
        self.retrieval_gt_chunk_type = chunk_type
        return []


@pytest.fixture
def fake_kovidore_rows() -> dict[tuple[str, str], list[dict[str, Any]]]:
    image = Image.new("RGB", (2, 2), color="white")
    return {
        ("queries", "test"): [
            {
                "query_id": 0,
                "query": "첫 번째 질문",
                "answer": "첫 번째 답변",
                "query_types": ["open-ended", "multi-hop"],
            },
            {
                "query_id": 1,
                "query": "두 번째 질문",
                "answer": "두 번째 답변",
                "query_types": ["open-ended"],
            },
        ],
        ("qrels", "test"): [
            {"query_id": 0, "corpus_id": 1, "score": 1},
            {"query_id": 0, "corpus_id": 2, "score": 2},
            {"query_id": 1, "corpus_id": 3, "score": 1},
            {"query_id": 1, "corpus_id": 4, "score": 0},
        ],
        ("corpus", "test"): [
            {
                "corpus_id": 0,
                "doc_id": "doc-a",
                "page_number_in_doc": 1,
                "markdown": "추가 문서",
                "elements": "[]",
                "modality": "image",
                "image": image,
            },
            {
                "corpus_id": 1,
                "doc_id": "doc-a",
                "page_number_in_doc": 2,
                "markdown": "첫 번째 근거",
                "elements": "[]",
                "modality": "image",
                "image": image,
            },
            {
                "corpus_id": 2,
                "doc_id": "doc-b",
                "page_number_in_doc": 1,
                "markdown": "두 번째 근거",
                "elements": "[]",
                "modality": "image",
                "image": image,
            },
            {
                "corpus_id": 3,
                "doc_id": "doc-c",
                "page_number_in_doc": 1,
                "markdown": "세 번째 근거",
                "elements": "[]",
                "modality": "image",
                "image": image,
            },
        ],
    }


def test_kovidorev2_detects_bigint_primary_key():
    ingestor = KoViDoReV2Ingestor("hr")

    assert ingestor.detect_primary_key_type() == "bigint"


def test_kovidorev2_is_registered_with_domain_choices():
    discover_ingestors.cache_clear()
    meta = discover_ingestors()["kovidorev2"]

    assert meta.ingestor_class is KoViDoReV2Ingestor
    dataset_param = next(param for param in meta.params if param.name == "dataset_name")
    assert dataset_param.choices == ["cybersecurity", "economic", "energy", "hr"]


def test_kovidorev2_ingests_beir_style_rows(monkeypatch, fake_kovidore_rows):
    def fake_load_dataset(_path: str, name: str, streaming: bool, split: str):
        assert streaming is (name == "corpus")
        return _FakeDataset(fake_kovidore_rows[(name, split)])

    monkeypatch.setattr("autorag_research.data.kovidorev2.load_dataset", fake_load_dataset)
    service = _FakeService()
    ingestor = KoViDoReV2Ingestor("hr")
    ingestor.set_service(service)  # type: ignore[arg-type]

    ingestor.ingest(min_corpus_cnt=4)

    assert [query["id"] for query in service.queries] == [0, 1]
    assert {chunk["id"] for chunk in service.image_chunks} == {
        0,
        1,
        2,
        3,
    }
    assert {chunk["id"] for chunk in service.chunks} == {
        0,
        1,
        2,
        3,
    }
    assert service.retrieval_gt_chunk_type == "image"
    assert len(service.retrieval_gt_items) == 2

    q0_gt = dict(service.retrieval_gt_items)[0]
    q0_relations = gt_to_relations(0, normalize_gt(q0_gt, chunk_type="image"))
    assert [relation["group_index"] for relation in q0_relations] == [0, 1]
    assert [relation["score"] for relation in q0_relations] == [1, 2]

    q1_gt = dict(service.retrieval_gt_items)[1]
    q1_relations = gt_to_relations(1, normalize_gt(q1_gt, chunk_type="image"))
    assert [relation["image_chunk_id"] for relation in q1_relations] == [3]


def test_kovidorev2_supports_mixed_qrels(monkeypatch, fake_kovidore_rows):
    def fake_load_dataset(_path: str, name: str, streaming: bool, split: str):
        return _FakeDataset(fake_kovidore_rows[(name, split)])

    monkeypatch.setattr("autorag_research.data.kovidorev2.load_dataset", fake_load_dataset)
    service = _FakeService()
    ingestor = KoViDoReV2Ingestor("hr", qrels_mode="mixed")
    ingestor.set_service(service)  # type: ignore[arg-type]

    ingestor.ingest(query_limit=1)

    assert service.retrieval_gt_chunk_type == "mixed"
    query_id, gt = service.retrieval_gt_items[0]
    relations = gt_to_relations(query_id, normalize_gt(gt, chunk_type="mixed"))
    assert query_id == 0
    assert len(relations) == 4
    assert {relation["group_index"] for relation in relations} == {0, 1}
    assert {relation["group_order"] for relation in relations} == {0, 1}


KOVIDOREV2_HR_CONFIG = IngestorTestConfig(
    expected_query_count=5,
    expected_chunk_count=20,
    expected_image_chunk_count=20,
    chunk_count_is_minimum=True,
    check_documents=True,
    check_pages=True,
    check_files=True,
    check_retrieval_relations=True,
    check_generation_gt=False,
    check_relevance_scores=True,
    primary_key_type="bigint",
    db_name="kovidorev2_hr_test",
)


@pytest.mark.data
class TestKoViDoReV2IngestorIntegration:
    """Integration tests using the real KoViDoRe V2 HR BEIR dataset."""

    def test_ingest_hr_subset(self):
        with create_test_database(KOVIDOREV2_HR_CONFIG) as db:
            service = MultiModalIngestionService(db.session_factory, schema=db.schema)  # ty: ignore[invalid-argument-type]

            ingestor = KoViDoReV2Ingestor("hr")
            ingestor.set_service(service)
            ingestor.ingest(
                query_limit=KOVIDOREV2_HR_CONFIG.expected_query_count,
                min_corpus_cnt=KOVIDOREV2_HR_CONFIG.expected_image_chunk_count,
            )

            verifier = IngestorTestVerifier(service, db.schema, KOVIDOREV2_HR_CONFIG)
            verifier.verify_all()
